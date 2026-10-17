"""Frozen DINOv2-B/14 feature cache extraction from already aligned RGB images."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn


class FrozenDINOv2(nn.Module):
    def __init__(self, repo, weights):
        super().__init__()
        # Local source and explicit checkpoint avoid silent network downloads.
        self.backbone = torch.hub.load(str(repo), 'dinov2_vitb14', source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        features = self.backbone.forward_features(images)
        return features['x_norm_clstoken'], features['x_norm_patchtokens']


def image_tensor(path):
    with Image.open(path) as image:
        x = np.array(image.convert('RGB').resize((224,224), Image.Resampling.BICUBIC), dtype=np.float32)/255
    x = torch.from_numpy(x).permute(2,0,1)
    return (x - torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]


def extract(aligned, output, repo, weights, device='cpu', batch_size=16):
    path = Path(aligned).resolve()
    with np.load(path, allow_pickle=False) as f:
        fixations, poses, frames = f['fixations_world'], f['world_camera'], f['frame_paths']
        segment_ids = f['segment_ids'] if 'segment_ids' in f.files else None
    if len(frames) != len(fixations) or len(poses) != len(frames) or not len(frames):
        raise ValueError('Need one pose and frame per fixation')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model = FrozenDINOv2(repo, weights).to(device)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing cache: {output}')
    output.mkdir(parents=True)
    # A directory of NPY arrays allows memory mapping during training.
    np.save(output/'fixations_world.npy', fixations)
    np.save(output/'world_camera.npy', poses)
    if segment_ids is not None:
        np.save(output/'segment_ids.npy', segment_ids)
    global_features = np.lib.format.open_memmap(output/'global_features.npy', mode='w+', dtype='float32', shape=(len(frames),768))
    dense_features = np.lib.format.open_memmap(output/'dense_features.npy', mode='w+', dtype='float32', shape=(len(frames),256,768))
    for start in range(0,len(frames),batch_size):
        images = torch.stack([image_tensor(path.parent/str(p)) for p in frames[start:start+batch_size]]).to(device)
        g, d = model(images)
        global_features[start:start+len(images)] = g.cpu().numpy()
        dense_features[start:start+len(images)] = d.cpu().numpy()
        print(f'Extracted {min(start+len(images),len(frames))}/{len(frames)} frames', flush=True)
    global_features.flush()
    dense_features.flush()
