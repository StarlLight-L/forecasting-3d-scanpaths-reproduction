"""Prepare every CDN recording sequentially, with verified caches and owned scratch."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scanpaths.data import validate_sequence


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def lock(root):
    with (root / '.pipeline.lock').open('a+b') as f:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not f.read(1):
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def make_plan(cdn, split, weights, threshold):
    if not cdn.get('sequences'):
        raise ValueError('Empty CDN sequence list')
    supplied_split = split is not None
    if split is None:
        split = {'unassigned': sorted(cdn['sequences'])}
    if supplied_split and set(split) != {'train', 'val', 'test'}:
        raise ValueError('Split file needs train, val and test lists')
    records, seen = [], set()
    for part, names in split.items():
        if not isinstance(names, list) or not names:
            raise ValueError('Each split must be a nonempty list')
        for name in names:
            if not isinstance(name, str) or '/' in name or '\\' in name or ':' in name or name in ('', '.', '..'):
                raise ValueError('Unsafe recording name')
            if name in seen:
                raise ValueError('Duplicate recording across splits')
            seen.add(name)
            assets = cdn['sequences'][name]
            records.append(dict(name=name, split=part, assets={
                k: {f: assets[k][f] for f in ('sha1sum', 'file_size_bytes')}
                for k in ('main_vrs', 'main_groundtruth')}))
    if seen != set(cdn['sequences']):
        raise ValueError('Split file must cover ALL CDN recordings exactly once; do not use the old 184-recording split')
    # Pin implementation as well as parameters; never mix caches after code changes.
    sources = ['scripts/export_adt.py', 'scripts/download_adt.py', 'scanpaths/data.py',
               'scanpaths/features.py', 'scanpaths/__main__.py']
    return dict(version=2, selection="all_cdn_sequences", records=records, threshold=threshold,
                weights_sha256=digest(weights), sources={s: digest(ROOT/s) for s in sources},
                exact_author_split=False)


def check_cache(cache, part):
    keys = ['fixations_world', 'world_camera', 'global_features', 'dense_features', 'segment_ids']
    arrays = {k: np.load(cache/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in keys}
    validate_sequence(arrays)
    n = len(arrays['fixations_world'])
    if arrays['global_features'].shape != (n, 768) or arrays['dense_features'].shape != (n, 256, 768):
        raise ValueError('Invalid DINOv2 feature shapes')
    ids = arrays['segment_ids']
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Invalid segment IDs')
    stride = 20 if part in ('val', 'test') else 1
    windows = sum(bool(np.all(ids[i:i+20] == ids[i])) for i in range(0, n-19, stride))
    if not windows:
        raise ValueError('No valid 10+10 windows; retaining raw files for investigation')
    return {k+'.npy': digest(cache/(k+'.npy')) for k in keys}


def verify(cache, record, plan):
    receipt = read(cache/'receipt.json')
    if receipt['record'] != record or receipt['plan'] != plan:
        raise ValueError('Cache provenance differs from plan')
    if receipt['files'] != check_cache(cache, record['split']):
        raise ValueError('Cache checksum mismatch; raw data retained')


def owned(work, name):
    marker = work/'.pipeline-owned'
    if work.is_symlink() or (work.exists() and (not marker.is_file() or marker.read_text() != name)):
        raise ValueError(f'Refusing unowned scratch: {work}')
    work.mkdir(parents=True, exist_ok=True)
    marker.write_text(name)


def process(record, plan, root, args):
    name = record['name']
    work, cache = root/'staging'/name, root/'features'/name
    if cache.is_symlink():
        raise ValueError('Refusing symlink cache')
    if cache.exists():
        verify(cache, record, plan)
    else:
        needed = sum(int(a['file_size_bytes']) for a in record['assets'].values()) + args.reserve_gb*1024**3
        if shutil.disk_usage(root).free < needed:
            raise OSError('Insufficient space for next download plus workspace reserve')
        owned(work, name)
        def run(*command):
            subprocess.run(list(map(str, command)), cwd=ROOT, check=True)
        run(sys.executable, ROOT/'scripts/download_adt.py', '--manifest', args.cdn_file,
            '--sequence', name, '--output', root/'staging')
        for folder in ('export', 'features.partial'):
            p = work/folder
            if p.is_symlink():
                raise ValueError('Refusing symlink scratch')
            if p.exists():
                shutil.rmtree(p)
        export, pending = work/'export', work/'features.partial'
        run(args.aria_python, ROOT/'scripts/export_adt.py', '--sequence', work, '--output', export)
        run(sys.executable, '-m', 'scanpaths', 'prepare', '--raw', export/'sequence.raw.npz',
            '--output', export/'sequence.aligned.npz', '--threshold', args.threshold)
        run(sys.executable, '-m', 'scanpaths', 'extract', '--aligned', export/'sequence.aligned.npz',
            '--output', pending, '--dino-repo', args.dino_repo, '--weights', args.weights, '--device', args.device)
        hashes = check_cache(pending, record['split'])
        save(pending/'receipt.json', dict(record=record, plan=plan, files=hashes))
        for p in pending.iterdir():
            with p.open('rb') as f:
                os.fsync(f.fileno())
        cache.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(cache)
        verify(cache, record, plan)
    if args.cleanup and work.exists():
        owned(work, name)
        shutil.rmtree(work)
    print(f'Verified: {name}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cdn-file', default='data/adt_download_urls-2.json')
    p.add_argument('--split-file', help='Optional explicit train/val/test lists covering ALL CDN recordings; no automatic split')
    p.add_argument('--output', default='data/adt_all_prepared')
    p.add_argument('--dino-repo', default='third_party/dinov2')
    p.add_argument('--weights', default='weights/dinov2_vitb14_pretrain.pth')
    p.add_argument('--aria-python', default=sys.executable)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--reserve-gb', type=float, default=20)
    p.add_argument('--limit', type=int, default=10, help='Maximum new recordings this run')
    p.add_argument('--all', action='store_true')
    p.add_argument('--cleanup', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.limit < 1 or not math.isfinite(args.threshold) or args.threshold <= 0 or not math.isfinite(args.reserve_gb) or args.reserve_gb < 0:
        p.error('Invalid limit, threshold or reserve')
    for key in ('cdn_file', 'split_file', 'dino_repo', 'weights', 'output'):
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)).resolve())
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        plan = make_plan(read(args.cdn_file), read(args.split_file) if args.split_file else None, args.weights, args.threshold)
        if (root/'plan.json').exists():
            if read(root/'plan.json') != plan:
                raise ValueError('Plan changed; use original split/settings/code or a fresh output directory')
        elif any(x.name != '.pipeline.lock' for x in root.iterdir()):
            raise ValueError('New output directory must be empty')
        save(root/'plan.json', plan)
        print(dict(sequences=len(plan['records']), splits={s: sum(r['split'] == s for r in plan['records'])
              for s in ('train','val','test','unassigned')}), flush=True)
        if args.plan_only:
            return
        if not (args.dino_repo/'hubconf.py').is_file():
            raise FileNotFoundError('DINOv2 source missing hubconf.py')
        completed, processed = [], 0
        for record in plan['records']:
            cache = root/'features'/record['name']
            fresh = not cache.exists()
            if fresh and not args.all and processed >= args.limit:
                continue
            process(record, plan, root, args)
            processed += int(fresh)
            completed.append(dict(path='features/'+record['name'], split=record['split']))
        save(root/'progress.json', dict(verified=len(completed), total=len(plan['records'])))
        if len(completed) == len(plan['records']):
            filename = 'manifest.json' if args.split_file else 'feature_manifest.json'
            save(root/filename, completed)
            print(f'All caches verified: {root / filename}')
            if not args.split_file:
                print('All recordings prepared. No train/val/test split assigned; do not train on this inventory.')
        else:
            print('Batch finished. Repeat the same command for the next batch.')


if __name__ == '__main__':
    main()
