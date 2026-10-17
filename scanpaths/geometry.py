"""SE(3) conventions: column vectors, T_world_camera, quaternion (x,y,z,w)."""
import numpy as np


def validate_transforms(t):
    t = np.asarray(t, dtype=np.float64)
    if t.shape[-2:] != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Expected finite [...,4,4] transforms")
    r = t[..., :3, :3]
    if not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError("Rotation must belong to SO(3)")
    if not np.allclose(t[..., 3, :], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous transform")
    return t


def rotation_to_quaternion(rotation):
    """Numerically stable eigenvector construction, including 180-degree rotations."""
    r = np.asarray(rotation)
    flat = r.reshape(-1, 3, 3)
    result = []
    for m in flat:
        k = np.array([
            [m[0,0]-m[1,1]-m[2,2], m[1,0]+m[0,1], m[2,0]+m[0,2], m[2,1]-m[1,2]],
            [m[1,0]+m[0,1], m[1,1]-m[0,0]-m[2,2], m[2,1]+m[1,2], m[0,2]-m[2,0]],
            [m[2,0]+m[0,2], m[2,1]+m[1,2], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
            [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], np.trace(m)]]) / 3
        _, vectors = np.linalg.eigh(k)
        q = vectors[:, -1]
        if q[3] < 0:
            q = -q
        result.append(q)
    return np.asarray(result).reshape(r.shape[:-2] + (4,))


def to_canonical(points_world, world_camera):
    t = validate_transforms(world_camera)
    return (np.asarray(points_world) - t[:3, 3]) @ t[:3, :3]


def to_world(points_canonical, world_camera):
    t = validate_transforms(world_camera)
    return np.asarray(points_canonical) @ t[:3, :3].T + t[:3, 3]


def canonical_poses(world_cameras, reference):
    ts, ref = validate_transforms(world_cameras), validate_transforms(reference)
    relative = np.linalg.inv(ref) @ ts
    return np.concatenate([rotation_to_quaternion(relative[:, :3, :3]), relative[:, :3, 3]], axis=-1)
