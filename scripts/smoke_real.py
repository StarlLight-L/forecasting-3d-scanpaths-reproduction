"""Single-sequence integration test, not a train/validation/test experiment."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scanpaths.data import ScanpathDataset
from scanpaths.model import ModelConfig, ScanpathTransformer, forecasting_loss
from scanpaths.engine import seed_everything, predict_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/adt_sample/manifest.json')
    p.add_argument('--output', default='runs/adt_sample')
    p.add_argument('--steps', type=int, default=30)
    a=p.parse_args()
    torch.set_num_threads(2)
    seed_everything(42)
    config=json.loads(Path('configs/paper.json').read_text())
    model=ScanpathTransformer(ModelConfig(**config['model']))
    dataset=ScanpathDataset(a.manifest,'train')
    batch=next(iter(DataLoader(dataset,batch_size=2,shuffle=False)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    losses=[]
    for step in range(a.steps):
        model.train()
        optimizer.zero_grad()
        prediction=predict_batch(model,batch,torch.device('cpu'))
        loss=forecasting_loss(prediction,batch['target'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        prediction=predict_batch(model,batch,torch.device('cpu'))
        final=float(forecasting_loss(prediction,batch['target']))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),config=config,epoch=0,smoke_only=True),output/'smoke.pt')
    report=dict(purpose='Single-recording integration/overfit test; no held-out evaluation',
                training_windows=len(dataset),batch_size=2,steps=a.steps,
                first_training_loss=losses[0],last_training_loss=losses[-1],final_batch_eval_mode_loss=final,
                prediction_shape=list(prediction.shape),finite_predictions=bool(torch.isfinite(prediction).all()),
                losses=losses)
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='losses'},indent=2))


if __name__=='__main__':
    main()
