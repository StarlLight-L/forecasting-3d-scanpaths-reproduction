import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.prepare_adt_batches import check_cache, digest, make_plan, process, save


def cache_fixture(tmp_path):
    root = tmp_path/'prepared'
    cache = root/'features'/'seq'
    cache.mkdir(parents=True)
    values = dict(fixations_world=np.ones((20, 4), dtype=np.float32),
                  world_camera=np.tile(np.eye(4), (20, 1, 1)),
                  global_features=np.zeros((20,768), dtype=np.float32),
                  dense_features=np.zeros((20,256,768), dtype=np.float32),
                  segment_ids=np.zeros(20, dtype=np.int64))
    for k,v in values.items():
        np.save(cache/(k+'.npy'), v)
    record, plan = dict(name='seq', split='test'), dict(version=1)
    save(cache/'receipt.json', dict(record=record, plan=plan, files=check_cache(cache,'test')))
    work=root/'staging'/'seq'
    work.mkdir(parents=True)
    (work/'.pipeline-owned').write_text('seq')
    (work/'video.vrs').write_bytes(b'raw source')
    return root, cache, work, record, plan


def test_verified_cache_cleanup_and_restart(tmp_path):
    root, cache, work, record, plan = cache_fixture(tmp_path)
    args=SimpleNamespace(cleanup=True)
    process(record, plan, root, args)
    assert not work.exists()
    assert (cache/'dense_features.npy').exists()
    process(record, plan, root, args)


def test_corrupt_cache_never_deletes_raw(tmp_path):
    root, cache, work, record, plan = cache_fixture(tmp_path)
    a=np.load(cache/'global_features.npy')
    a[0,0]=1
    np.save(cache/'global_features.npy', a)
    with pytest.raises(ValueError, match='checksum'):
        process(record, plan, root, SimpleNamespace(cleanup=True))
    assert (work/'video.vrs').exists()


def test_unowned_scratch_never_deleted(tmp_path):
    root, cache, work, record, plan = cache_fixture(tmp_path)
    (work/'.pipeline-owned').unlink()
    with pytest.raises(ValueError, match='unowned'):
        process(record, plan, root, SimpleNamespace(cleanup=True))
    assert (work/'video.vrs').exists()


def test_reject_cache_without_evaluation_windows(tmp_path):
    root, cache, work, record, plan = cache_fixture(tmp_path)
    np.save(cache/'segment_ids.npy', np.arange(20))
    with pytest.raises(ValueError, match='windows'):
        check_cache(cache,'val')


def test_plan_pins_split_and_ignores_url_refresh(tmp_path):
    weights=tmp_path/'weights'
    weights.write_bytes(b'weights')
    asset=dict(sha1sum='abc', file_size_bytes=100, download_url='old')
    cdn={'sequences':{n:{k:dict(asset) for k in ('main_vrs','main_groundtruth')} for n in ('a','b','c')}}
    split=dict(train=['a'],val=['b'],test=['c'])
    plan=make_plan(cdn,split,weights,0.05)
    cdn['sequences']['a']['main_vrs']['download_url']='new'
    assert make_plan(cdn,split,weights,0.05)==plan
    assert make_plan(cdn,split,weights,0.1)!=plan
    with pytest.raises(ValueError, match='Duplicate'):
        make_plan(cdn,dict(train=['a'],val=['a'],test=['c']),weights,0.05)
