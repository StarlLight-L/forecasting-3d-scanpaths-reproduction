import json
import numpy as np
from scanpaths.data import ScanpathDataset


def dataset(tmp_path, durations, ids, timestamps=None):
    n=len(durations)
    values=dict(fixations_world=np.column_stack([np.zeros((n,3)), durations]),
                world_camera=np.tile(np.eye(4), (n,1,1)),
                global_features=np.zeros((n,8)),dense_features=np.zeros((n,4,8)),
                segment_ids=np.array(ids,dtype=np.int64))
    if timestamps is not None:
        values['fixation_start_times']=timestamps
        values['fixation_end_times']=timestamps+durations
    np.savez(tmp_path/'cache.npz',**values)
    m=tmp_path/'manifest.json'
    m.write_text(json.dumps([dict(path='cache.npz',split='test')]))
    return m


def test_shared_eval_restarts_at_valid_candidate(tmp_path):
    m=dataset(tmp_path,np.full(45,.1),[0]*5+[1]*40)
    shared=ScanpathDataset(m,'test',window_protocol='scanpath1')
    assert shared.index==[(0,5),(0,25)]
    assert ScanpathDataset(m,'test').index==[(0,20)]


def test_shared_filters_long_windows_and_advances_stride_after_acceptance(tmp_path):
    durations=np.full(41,.1);durations[0]=12
    m=dataset(tmp_path,durations,[0]*41)
    assert ScanpathDataset(m,'test',window_protocol='scanpath1').index==[(0,1),(0,21)]


def test_shared_uses_elapsed_timestamps(tmp_path):
    durations=np.full(41,.1)
    times=np.arange(41)*.1;times[10:]+=12
    m=dataset(tmp_path,durations,[0]*41,times)
    assert ScanpathDataset(m,'test',window_protocol='scanpath1').index==[(0,10)]
