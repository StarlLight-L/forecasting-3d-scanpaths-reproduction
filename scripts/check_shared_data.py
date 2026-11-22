"""Read-only audit of a shared Scanpath1 manifest and Ryan input windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanpaths.data import ScanpathDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--expected-sequences', type=int, default=236)
    a = p.parse_args()
    path = Path(a.manifest).resolve()
    items = json.loads(path.read_text())
    counts = Counter(e['split'] for e in items)
    print(dict(sequences=len(items), splits=dict(counts)), flush=True)
    if len(items) != a.expected_sequences or set(counts) != {'train','val','test'}:
        raise ValueError('Unexpected recording count or missing train/val/test split')
    families = {}
    for e in items:
        name = e.get('recording_id', Path(e['path']).name)
        family = re.sub(r'_\d+s_sample(?=_|$)', '', name)
        if family in families and families[family] != e['split']:
            raise ValueError('Sample and parent recording cross splits')
        families[family] = e['split']
        cache = path.parent/e['path']
        meta = json.loads((cache/'preprocessing.json').read_text())
        if meta.get('dino') != 'dinov2_vitb14' or meta.get('image_size') != 224:
            raise ValueError(f'Unexpected DINO provenance: {cache}')
        for key in ('segment_ids', 'fixation_start_times', 'fixation_end_times'):
            if not (cache/(key+'.npy')).is_file():
                raise ValueError(f'Missing shared-cache metadata: {cache}/{key}.npy')
    for split in ('train','val','test'):
        data = ScanpathDataset(path, split, window_protocol='scanpath1')
        for s in data.sequences:
            n = len(s['fixations_world'])
            if s['global_features'].shape != (n,768) or s['dense_features'].shape != (n,256,768):
                raise ValueError('Unexpected DINO feature dimensions')
        print(f'{split}: {len(data)} valid windows', flush=True)
        del data
    print('Shared-cache audit passed; no files changed.')


if __name__ == '__main__':
    main()
