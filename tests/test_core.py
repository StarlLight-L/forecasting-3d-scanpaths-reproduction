import json
import numpy as np
import pytest
import torch
from scanpaths.model import ModelConfig, ScanpathTransformer, forecasting_loss
from scanpaths.geometry import to_canonical, to_world, canonical_poses, rotation_to_quaternion
from scanpaths.data import ScanpathDataset, cluster_fixations
from scanpaths.metrics import distances, baseline

torch.set_num_threads(2)


def test_geometry_rotated_translated_reference():
    t = np.eye(4)
    t[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    t[:3,3] = [10,20,30]
    world = np.array([[10,21,30],[9,20,30.]])
    local = to_canonical(world,t)
    np.testing.assert_allclose(local,[[1,0,0],[0,1,0]],atol=1e-6)
    np.testing.assert_allclose(to_world(local,t),world)
    np.testing.assert_allclose(canonical_poses(t[None],t),[[0,0,0,1,0,0,0]],atol=1e-6)
    np.testing.assert_allclose(rotation_to_quaternion(t[:3,:3]),[0,0,np.sqrt(.5),np.sqrt(.5)],atol=1e-6)
    q = rotation_to_quaternion(np.diag([1,-1,-1]))
    np.testing.assert_allclose(np.abs(q),[1,0,0,0],atol=1e-6)


def test_distance_known_answer():
    a = np.zeros((10,4)); b = a.copy(); b[:,0] = 2
    assert distances(a,a)['FRE'] == 0
    for key in ['EUC','DTW','FRE']:
        assert distances(a,b)[key] == 2
    o = np.array([[0.,0,0,1],[1,0,0,1]])
    np.testing.assert_allclose(baseline(o,3,'linear')[:,0],[2,3,4])


def test_clustering_breaks_missing_samples():
    p = np.array([[0.,0,0],[.01,0,0],[np.nan,0,0],[1,0,0],[1.01,0,0]])
    f, ends = cluster_fixations(p,np.arange(5)/30,.05)
    np.testing.assert_array_equal(ends,[1,4])
    np.testing.assert_allclose(f[:,3],2/30)


def write_sequence(tmp_path):
    n = 12
    s = dict(fixations_world=np.c_[np.arange(n),np.zeros((n,2)),np.ones(n)].astype('float32'),
             world_camera=np.tile(np.eye(4),(n,1,1)),
             global_features=np.ones((n,8)),dense_features=np.ones((n,4,8)))
    np.savez(tmp_path/'seq.npz',**s)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps([dict(path='seq.npz',split='train')]))
    return s,path


def test_window_split_and_no_future_leakage(tmp_path):
    s,path = write_sequence(tmp_path)
    ds = ScanpathDataset(path,'train',3,2)
    before = ds[0]
    assert len(ds) == 8
    ds.sequences[0]['fixations_world'][3:] += 100
    ds.sequences[0]['world_camera'][3:,:3,3] += 100
    ds.sequences[0]['dense_features'][3:] += 100
    ds.sequences[0]['global_features'][3:] += 100
    after = ds[0]
    for key in ['observed','poses','dense_features','global_features']:
        torch.testing.assert_close(before[key],after[key])
    assert not torch.equal(before['target'],after['target'])
    path.write_text(json.dumps([dict(path='seq.npz',split='train'),dict(path='seq.npz',split='test')]))
    with pytest.raises(ValueError,match='Duplicate'):
        ScanpathDataset(path,'train',3,2)


@pytest.mark.parametrize('mode',['bidirectional','partial','causal'])
@pytest.mark.parametrize('trajectory_only',[False,True])
def test_forward_backward_ablation(mode,trajectory_only):
    c = ModelConfig(observed=3,future=2,feature_dim=8,dim=16,heads=4,ff_dim=32,dropout=0,attention=mode,trajectory_only=trajectory_only)
    model = ScanpathTransformer(c)
    result = model(torch.randn(2,3,4),torch.randn(2,2,8),torch.randn(2,4,8),torch.randn(2,3,7))
    assert result.shape == (2,2,4)
    forecasting_loss(result,torch.zeros_like(result)).backward()
    assert model.queries.grad is not None and torch.isfinite(model.queries.grad).all()
    assert model.fixation.weight.grad.abs().sum() > 0
    if not trajectory_only:
        for layer in [model.dense,model.global_feature,model.pose]:
            assert layer.weight.grad.abs().sum() > 0


def test_small_batch_can_overfit():
    torch.manual_seed(3)
    model = ScanpathTransformer(ModelConfig(observed=3,future=2,feature_dim=8,dim=16,heads=4,ff_dim=32,dropout=0))
    inputs = [torch.randn(2,3,4),torch.randn(2,2,8),torch.randn(2,4,8),torch.randn(2,3,7)]
    target = torch.randn(2,2,4)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.005)
    first = forecasting_loss(model(*inputs),target).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = forecasting_loss(model(*inputs),target)
        loss.backward(); optimizer.step()
    assert forecasting_loss(model(*inputs),target).item() < first*.1


def test_memory_mapped_cache_matches_npz(tmp_path):
    s,path = write_sequence(tmp_path)
    expected = ScanpathDataset(path,'train',3,2)[0]
    directory = tmp_path/'cache'
    directory.mkdir()
    for key,value in s.items():
        np.save(directory/(key+'.npy'),value)
    path.write_text(json.dumps([dict(path='cache',split='train')]))
    dataset = ScanpathDataset(path,'train',3,2)
    assert isinstance(dataset.sequences[0]['dense_features'],np.memmap)
    for key,value in dataset[0].items():
        torch.testing.assert_close(value,expected[key])


def test_frozen_backbone_stays_eval(tmp_path, monkeypatch):
    from scanpaths.features import FrozenDINOv2

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.dropout = torch.nn.Dropout(.9)

        def forward_features(self, x):
            y = self.dropout(x) * self.weight
            return {'x_norm_clstoken': y[:,0], 'x_norm_patchtokens': y}

    torch.save(Backbone().state_dict(), tmp_path/'weights.pt')
    monkeypatch.setattr(torch.hub, 'load', lambda *a, **kw: Backbone())
    model = FrozenDINOv2(tmp_path,tmp_path/'weights.pt')
    model.train()
    x = torch.ones(2,3,4,requires_grad=True)
    g,d = model(x)
    assert not model.backbone.training
    assert not any(p.requires_grad for p in model.parameters())
    assert not d.requires_grad
    torch.testing.assert_close(d,x)


def test_windows_do_not_cross_missing_data(tmp_path):
    from scanpaths.data import fixation_segment_ids
    s,path = write_sequence(tmp_path)
    s['segment_ids'] = np.r_[np.zeros(6,dtype=int),np.ones(6,dtype=int)]
    np.savez(tmp_path/'seq.npz',**s)
    dataset=ScanpathDataset(path,'train',3,2)
    assert [start for _,start in dataset.index] == [0,1,6,7]
    points=np.array([[0.,0,0],[0,0,0],[np.nan,0,0],[1,0,0],[1,0,0]])
    ids=fixation_segment_ids(points,np.arange(5)/30,np.array([1,4]))
    assert ids[0] != ids[1]
