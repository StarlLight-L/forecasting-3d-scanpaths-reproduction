import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .engine import train, load_model, choose_device, evaluate, predict_batch
from .data import ScanpathDataset, cluster_fixations, fixation_segment_ids
from .geometry import to_world


def main():
    parser = argparse.ArgumentParser(description='Egocentric 3D scanpath reproduction')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train')
    p.add_argument('--config', default='configs/paper.json')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='runs/paper')
    p.add_argument('--device', default='auto')
    for command in ['evaluate','predict']:
        p = commands.add_parser(command)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--manifest', required=True)
        p.add_argument('--split', default='test')
        p.add_argument('--device', default='auto')
        p.add_argument('--output', required=True)
        if command == 'evaluate':
            p.add_argument('--baseline', choices=['last','mean','linear'])
    p = commands.add_parser('extract')
    p.add_argument('--aligned', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dino-repo', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='cpu')
    p = commands.add_parser('prepare')
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', required=True, type=float)
    p.add_argument('--sample-period', default=1/30, type=float)
    args = parser.parse_args()
    if args.command == 'train':
        train(json.loads(Path(args.config).read_text()), args.manifest, args.output, args.device)
    elif args.command == 'extract':
        from .features import extract
        extract(args.aligned, args.output, args.dino_repo, args.weights, args.device)
    elif args.command == 'prepare':
        raw = Path(args.raw).resolve()
        with np.load(raw, allow_pickle=False) as s:
            fixations, indices = cluster_fixations(s['gaze_world'], s['timestamps'], args.threshold, args.sample_period)
            frames = np.array([str((raw.parent/str(p)).resolve()) for p in s['frame_paths'][indices]])
            poses = s['world_camera'][indices]
            segment_ids = fixation_segment_ids(s['gaze_world'], s['timestamps'], indices, args.sample_period)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, fixations_world=fixations, world_camera=poses, frame_paths=frames, segment_ids=segment_ids)
    else:
        device = choose_device(args.device)
        model = load_model(args.checkpoint, device)
        c = model.config
        loader = DataLoader(ScanpathDataset(args.manifest,args.split,c.observed,c.future), batch_size=16)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'evaluate':
            result = evaluate(model, loader, device, args.baseline)
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        else:
            canonical, world, transforms = [], [], []
            with torch.no_grad():
                for batch in loader:
                    pred = predict_batch(model,batch,device).cpu().numpy()
                    canonical.extend(pred)
                    transforms.extend(batch['world_camera'].numpy())
                    for point, transform in zip(pred,batch['world_camera'].numpy()):
                        converted = point.copy()
                        converted[:,:3] = to_world(point[:,:3],transform)
                        world.append(converted)
            np.savez_compressed(output, canonical=np.array(canonical), world=np.array(world), world_camera=np.array(transforms))


if __name__ == '__main__':
    main()
