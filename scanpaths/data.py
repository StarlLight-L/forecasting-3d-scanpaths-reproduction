"""Fixation-level sequence caches. Windows never cross recording/split boundaries."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import to_canonical, canonical_poses, validate_transforms


class ScanpathDataset(Dataset):
    def __init__(self, manifest, split, observed=10, future=10, stride=None, window_protocol="legacy", max_window_seconds=10.0):
        if window_protocol not in ('legacy', 'scanpath1'):
            raise ValueError('Unknown window protocol')
        if max_window_seconds <= 0 or not np.isfinite(max_window_seconds):
            raise ValueError('Invalid maximum window duration')
        self.observed, self.future = observed, future
        path = Path(manifest).resolve()
        items = json.loads(path.read_text())
        seen = set()
        self.sequences, self.index = [], []
        if observed < 1 or future < 1:
            raise ValueError("Sequence lengths must be positive")
        stride = (1 if split == "train" else observed + future) if stride is None else stride
        if stride < 1:
            raise ValueError("stride must be positive")
        for entry in items:
            source = (path.parent / entry['path']).resolve()
            if source in seen:
                raise ValueError(f"Duplicate recording across manifest entries: {source}")
            seen.add(source)
            if 'target_range' in entry:
                raise ValueError('Temporal target_range manifests are not supported by this Ryan loader')
            if entry['split'] != split:
                continue
            if source.is_dir():
                data = {k: np.load(source / (k + '.npy'), mmap_mode='r', allow_pickle=False)
                        for k in ['fixations_world', 'world_camera', 'global_features', 'dense_features']}
                for key in ('segment_ids', 'fixation_start_times', 'fixation_end_times'):
                    if (source/(key+'.npy')).exists():
                        data[key] = np.load(source/(key+'.npy'), allow_pickle=False)
            else:
                with np.load(source, allow_pickle=False) as f:
                    data = {k: f[k] for k in f.files}
            validate_sequence(data)
            seq_id = len(self.sequences)
            self.sequences.append(data)
            if window_protocol == 'scanpath1':
                next_start = 0
                for start in shared_window_starts(data, observed + future, max_window_seconds):
                    if start >= next_start:
                        self.index.append((seq_id, start))
                        next_start = start + stride
            else:
                for start in range(0, len(data['fixations_world']) - observed - future + 1, stride):
                    ids = data.get('segment_ids')
                    if ids is None or np.all(ids[start:start+observed+future] == ids[start]):
                        self.index.append((seq_id, start))
        if not self.index:
            raise ValueError(f"No complete windows in split {split!r}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        seq, start = self.index[index]
        s = self.sequences[seq]
        end = start + self.observed
        reference = s['world_camera'][end - 1]
        fixations = s['fixations_world'][start:end + self.future].copy()
        fixations[:, :3] = to_canonical(fixations[:, :3], reference)
        values = dict(observed=fixations[:self.observed], target=fixations[self.observed:],
                      poses=canonical_poses(s['world_camera'][start:end], reference),
                      global_features=s['global_features'][start:end - 1],
                      dense_features=s['dense_features'][end - 1], world_camera=reference)
        return {key: torch.from_numpy(np.asarray(value, dtype=np.float32).copy()) for key, value in values.items()}


def validate_sequence(s):
    required = {'fixations_world', 'world_camera', 'global_features', 'dense_features'}
    if not required <= s.keys():
        raise ValueError(f"Missing keys: {required - s.keys()}")
    n = len(s['fixations_world'])
    if s['fixations_world'].shape != (n, 4) or s['world_camera'].shape != (n, 4, 4):
        raise ValueError("Invalid fixation or pose shape")
    g, d = s['global_features'], s['dense_features']
    if g.ndim != 2 or d.ndim != 3 or len(g) != n or len(d) != n or g.shape[-1] != d.shape[-1] or d.shape[1] < 1:
        raise ValueError("Expected global [N,D] and dense [N,P,D] features")
    # Chunk validation so large memory-mapped patch caches do not allocate N*P*D booleans.
    finite = all(np.isfinite(s[k][start:start+128]).all()
                 for k in required for start in range(0, n, 128))
    if not finite or (s['fixations_world'][:, 3] <= 0).any():
        raise ValueError("Non-finite values or non-positive durations")
    validate_transforms(s['world_camera'])
    if 'segment_ids' in s:
        ids = s['segment_ids']
        if ids.shape != (n,) or ids.dtype.kind not in 'iu' or (np.diff(ids) < 0).any():
            raise ValueError('segment_ids must be an ordered integer vector of length N')
    timing = ('fixation_start_times', 'fixation_end_times')
    if any(k in s for k in timing):
        if not all(k in s for k in timing):
            raise ValueError('Both fixation timestamp arrays are required')
        a, b = (s[k] for k in timing)
        if (a.shape != (n,) or b.shape != (n,) or not np.isfinite(a).all()
                or not np.isfinite(b).all() or (b <= a).any()
                or (np.diff(a) <= 0).any() or (b[:-1] > a[1:] + 1e-6).any()):
            raise ValueError('Invalid fixation timestamp intervals')



def cluster_fixations(points, timestamps, threshold, sample_period=1/30):
    """Explicit approximation: adjacent-point distance clustering, no minimum duration.

    Invalid samples break clusters. Each fixation uses its mean location and last
    valid frame. Threshold is deliberately required: not specified in main paper.
    """
    points, timestamps = np.asarray(points), np.asarray(timestamps)
    if points.shape != (len(timestamps), 3) or threshold <= 0 or sample_period <= 0:
        raise ValueError("Invalid shape or non-positive clustering parameters")
    if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    clusters, group = [], []
    for i, point in enumerate(points):
        valid = np.isfinite(point).all()
        if group and (not valid or np.linalg.norm(point - points[group[-1]]) > threshold or timestamps[i] - timestamps[group[-1]] > 1.5 * sample_period):
            clusters.append(group)
            group = []
        if valid:
            group.append(i)
    if group:
        clusters.append(group)
    fixation = [np.r_[points[g].mean(0), timestamps[g[-1]] - timestamps[g[0]] + sample_period] for g in clusters]
    return np.asarray(fixation, dtype=np.float32).reshape(-1, 4), np.array([g[-1] for g in clusters], dtype=np.int64)


def fixation_segment_ids(points, timestamps, ends, sample_period=1/30):
    """Retain missing-data boundaries after invalid samples are removed by clustering."""
    valid = np.isfinite(points).all(axis=1)
    breaks = np.ones(len(valid), dtype=bool)
    if len(valid)>1:
        breaks[1:] = (~valid[1:]) | (~valid[:-1]) | (np.diff(timestamps)>1.5*sample_period)
    return np.cumsum(breaks)[ends].astype(np.int64)


def shared_window_starts(data, length, max_seconds=10.0):
    """Match Scanpath1's contiguous-run, elapsed-time candidate enumeration."""
    n = len(data['fixations_world'])
    ids = data.get('segment_ids', np.zeros(n, dtype=np.int64))
    bounds = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, n]
    cumulative = np.r_[0., np.cumsum(data['fixations_world'][:, 3], dtype=np.float64)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        for start in range(int(left), int(right)-length+1):
            stop = start + length
            seconds = (data['fixation_end_times'][stop-1] - data['fixation_start_times'][start]
                       if 'fixation_end_times' in data else cumulative[stop]-cumulative[start])
            if seconds <= max_seconds + 1e-6:
                yield start
