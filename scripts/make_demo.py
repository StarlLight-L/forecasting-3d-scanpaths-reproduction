"""Generate synthetic caches for smoke tests only; NOT ADT or DINO features."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--output', default='data/demo')
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
entries = []
for split in ['train','val','test']:
    n = 40
    points = np.cumsum(rng.normal(0,.01,(n,3)),0) + [0,0,2]
    fixation = np.c_[points,np.full(n,.15)].astype('float32')
    poses = np.tile(np.eye(4),(n,1,1)).astype('float32')
    poses[:,0,3] = np.arange(n)*.005
    np.savez_compressed(out/f'{split}.npz',fixations_world=fixation,world_camera=poses,
                        global_features=rng.normal(size=(n,768)).astype('float32'),
                        dense_features=rng.normal(size=(n,256,768)).astype('float32'))
    entries.append(dict(path=f'{split}.npz',split=split))
(out/'manifest.json').write_text(json.dumps(entries,indent=2))
print(out/'manifest.json')
