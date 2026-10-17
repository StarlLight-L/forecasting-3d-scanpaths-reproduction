import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ModelConfig, ScanpathTransformer, forecasting_loss
from .data import ScanpathDataset
from .metrics import distances, baseline


INPUTS = ('observed', 'global_features', 'dense_features', 'poses')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_batch(model, batch, device):
    return model(**{k: batch[k].to(device) for k in INPUTS})


@torch.no_grad()
def evaluate(model, loader, device, baseline_kind=None):
    model.eval()
    values = []
    for batch in loader:
        prediction = (np.stack([baseline(o.numpy(), batch['target'].shape[1], baseline_kind)
                                for o in batch['observed']]) if baseline_kind
                      else predict_batch(model, batch, device).cpu().numpy())
        values.extend(distances(p, t) for p, t in zip(prediction, batch['target'].numpy()))
    return {k: float(np.mean([v[k] for v in values])) for k in values[0]}


def train(config, manifest, output, device='auto'):
    seed_everything(config['training']['seed'])
    device = choose_device(device)
    c, settings = ModelConfig(**config['model']), config['training']
    datasets = {split: ScanpathDataset(manifest, split, c.observed, c.future) for split in ['train','val']}
    loaders = {split: DataLoader(data, batch_size=settings['batch_size'], shuffle=split=='train') for split,data in datasets.items()}
    model = ScanpathTransformer(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    history = []
    for epoch in range(settings['epochs']):
        model.train()
        total = 0.0
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_batch(model, batch, device)
            loss = forecasting_loss(prediction, batch['target'].to(device), settings['position_weight'], settings['duration_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(prediction)
        validation = evaluate(model, loaders['val'], device)
        record = dict(epoch=epoch+1, train_loss=total/len(datasets['train']), validation=validation)
        history.append(record)
        print(json.dumps(record), flush=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, epoch=epoch+1)
        torch.save(state, output/'last.pt')
        if validation['EUC'] < best:
            best = validation['EUC']
            torch.save(state, output/'best.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    return history


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = ScanpathTransformer(ModelConfig(**state['config']['model'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()
