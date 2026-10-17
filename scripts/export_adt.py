"""Export aligned native-orientation RGB, GT gaze and RGB poses from one ADT VRS.

Run in .venv-adt (Python 3.12); output is consumed by the training environment.
No estimated constant gaze depth is substituted for missing GT depth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId


def nearest_index(times, timestamp):
    i=int(np.searchsorted(times,timestamp))
    return min([j for j in [i-1,i] if 0 <= j < len(times)],key=lambda j:abs(int(times[j])-timestamp))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sequence',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-gaze-dt-ms',type=float,default=20)
    p.add_argument('--max-pose-dt-ms',type=float,default=5)
    a=p.parse_args()
    root=Path(a.sequence)
    output=Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    frames=output/'frames'
    frames.mkdir(exist_ok=True)
    provider=data_provider.create_vrs_data_provider(str(root/'video.vrs'))
    if provider is None:
        raise RuntimeError('VRS provider could not open recording')
    stream=StreamId('214-1')
    label=provider.get_label_from_stream_id(stream)
    calib=provider.get_device_calibration()
    device_camera=calib.get_transform_device_sensor(label).to_matrix()
    device_cpf=calib.get_transform_device_cpf().to_matrix()
    gaze=mps.read_eyegaze(str(root/'eyegaze.csv'))
    trajectory=mps.read_closed_loop_trajectory(str(root/'aria_trajectory.csv'))
    gaze_times=np.array([round(g.tracking_timestamp.total_seconds()*1e9) for g in gaze],dtype=np.int64)
    pose_times=np.array([round(t.tracking_timestamp.total_seconds()*1e9) for t in trajectory],dtype=np.int64)
    if len(gaze_times)==0 or len(pose_times)==0 or (np.diff(gaze_times)<0).any() or (np.diff(pose_times)<0).any():
        raise ValueError('Empty or unsorted ground truth')
    timestamps=np.array(provider.get_timestamps_ns(stream,TimeDomain.DEVICE_TIME),dtype=np.int64)
    points=[]; poses=[]; paths=[]; gaze_dt=[]; pose_dt=[]
    valid=0
    for index,timestamp in enumerate(timestamps):
        timestamp=int(timestamp)
        gi=nearest_index(gaze_times,timestamp)
        pi=nearest_index(pose_times,timestamp)
        gd=abs(int(gaze_times[gi])-timestamp)
        pd=abs(int(pose_times[pi])-timestamp)
        g=gaze[gi]
        world_device=trajectory[pi].transform_world_device.to_matrix()
        poses.append(world_device@device_camera)
        # Gaze lives at its own timestamp; transform it using that time's device pose.
        gaze_pi=nearest_index(pose_times,int(gaze_times[gi]))
        good=(gd<=a.max_gaze_dt_ms*1e6 and pd<=a.max_pose_dt_ms*1e6
              and abs(int(pose_times[gaze_pi])-int(gaze_times[gi]))<=a.max_pose_dt_ms*1e6
              and np.isfinite([g.depth,g.yaw,g.pitch]).all() and g.depth>0)
        point=np.full(3,np.nan)
        if good:
            cpf=mps.get_eyegaze_point_at_depth(g.yaw,g.pitch,g.depth)
            world_cpf=trajectory[gaze_pi].transform_world_device.to_matrix()@device_cpf
            point=(world_cpf@np.r_[cpf,1])[:3]
            valid+=1
        points.append(point)
        image,_=provider.get_image_data_by_index(stream,index)
        image_path=frames/f'{index:06d}.png'
        Image.fromarray(image.to_numpy_array()).convert('RGB').resize((224,224),Image.Resampling.BICUBIC).save(image_path)
        paths.append(str(image_path.relative_to(output)))
        gaze_dt.append(gd);pose_dt.append(pd)
    np.savez_compressed(output/'sequence.raw.npz',gaze_world=np.array(points),
                        timestamps=timestamps.astype(np.float64)/1e9,world_camera=np.array(poses),frame_paths=np.array(paths))
    report=dict(sequence=root.name,frames=len(timestamps),valid_gaze_frames=valid,
                invalid_gaze_frames=len(timestamps)-valid,
                duration_seconds=float((timestamps[-1]-timestamps[0])/1e9),
                max_gaze_alignment_ms=float(max(gaze_dt)/1e6),max_pose_alignment_ms=float(max(pose_dt)/1e6),
                world_pose_source='ADT ground-truth aria_trajectory.csv (not MPS SLAM)',
                gaze_source='ADT ground-truth eyegaze.csv depth_m',
                camera_convention='Native unrotated RGB; T_world_camera = T_world_device @ T_device_camera',
                image_processing='RGB, bicubic 224x224, no rotation',
                timestamp_policy='nearest with explicit tolerance; gaze transformed at gaze timestamp',
                device_camera=device_camera.tolist(),device_cpf=device_cpf.tolist())
    (output/'export_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['device_camera','device_cpf']},indent=2))


if __name__=='__main__':
    main()
