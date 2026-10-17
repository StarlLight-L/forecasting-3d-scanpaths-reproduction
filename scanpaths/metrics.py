"""Unambiguous distance metrics; other paper metrics are intentionally not aliased."""
import numpy as np


def distances(prediction, target):
    a, b = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] < 3 or not len(a):
        raise ValueError("Expected equal, nonempty [N,>=3] scanpaths")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scanpaths must be finite")
    cost = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=-1)
    n = len(a)
    dtw = np.full((n+1, n+1), np.inf)
    fre = np.full_like(dtw, np.inf)
    dtw[0, 0] = fre[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
            fre[i, j] = max(cost[i-1,j-1], min(fre[i-1,j], fre[i,j-1], fre[i-1,j-1]))
    result = {'EUC': float(cost.diagonal().mean()), 'DTW': float(dtw[n,n]/n), 'FRE': float(fre[n,n])}
    if a.shape[1] == 4:
        result['duration_MAE_seconds'] = float(np.abs(a[:,3]-b[:,3]).mean())
    return result


def baseline(observed, future, kind='last'):
    """Deterministic position baselines; duration persistence is our extension."""
    observed = np.asarray(observed)
    if kind == 'last':
        return np.repeat(observed[-1:], future, axis=0)
    if kind == 'mean':
        return np.repeat(observed.mean(0, keepdims=True), future, axis=0)
    if kind == 'linear':
        if len(observed) < 2:
            raise ValueError("Linear baseline needs two observations")
        result = np.repeat(observed[-1:], future, axis=0)
        result[:, :3] += np.arange(1, future+1)[:,None] * (observed[-1,:3]-observed[-2,:3])
        return result
    raise ValueError(kind)
