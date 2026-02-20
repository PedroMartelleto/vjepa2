# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from pathlib import Path
import rerun as rr
import hydra
import json
import numpy as np

from vipe.streams.droid import DecordDroidStream
from vipe.pipeline.droid import DroidPreprocessingPipeline

def parse_intrinsics_simple(video_path: Path):
    """Simple heuristic to find metadata json and get intrinsics"""
    # Go up directories to find metadata
    current = video_path.parent
    for _ in range(4): # Try 4 levels up
        metas = list(current.glob("metadata_*.json"))
        if metas:
            meta_path = metas[0]
            with open(meta_path, 'r') as f:
                data = json.load(f)
            
            serial = video_path.stem
            # Simple list check (DROID standard)
            if "cameras" in data:
                for cam in data["cameras"]:
                    if str(cam.get("serial_number")) == serial or str(cam.get("serial")) == serial:
                        if "intrinsics" in cam: return np.array(cam["intrinsics"])
            break
        current = current.parent
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output", type=Path, default="debug_droid.rrd")
    args = parser.parse_args()
    
    intrinsics = parse_intrinsics_simple(args.video_path)
    if intrinsics is None:
        print(f"Error: Could not find metadata json or intrinsics for {args.video_path}")
        return
    
    print(f"Found Intrinsics: {intrinsics}")

    rr.init("droid_sanity_check", spawn=False)
    rr.save(str(args.output))

    overrides = [
        "pipeline.instance=vipe.pipeline.droid.DroidPreprocessingPipeline",
        "pipeline.slam.keyframe_depth=unidepth-l", 
        "pipeline.post.depth_align_model=adaptive_unidepth-l_svda",
        "pipeline.slam.optimize_intrinsics=false" 
    ]
    
    with hydra.initialize_config_dir(config_dir="../configs", version_base=None):
        cfg = hydra.compose("default", overrides=overrides)

    stream = DecordDroidStream(args.video_path, intrinsics)
    pipeline = DroidPreprocessingPipeline(
        init=cfg.pipeline.init,
        slam=cfg.pipeline.slam,
        post=cfg.pipeline.post,
        output=cfg.pipeline.output
    )
    
    pipeline.return_output_streams = True
    print("Running Pipeline...")
    output = pipeline.run(stream)
    final_stream = output.output_streams[0]

    for frame in final_stream:
        rr.set_time_sequence("frame", frame.raw_frame_idx)
        rr.log("camera/image", rr.Image(frame.rgb.cpu().numpy()))
        
        t = frame.pose.translation().cpu().numpy()
        q_xyzw = frame.pose.quaternion().cpu().numpy()
        
        rr.log("camera", rr.Transform3D(
            translation=t,
            rotation=rr.Quaternion(xyzw=q_xyzw),
            from_parent=False
        ))
        
        if hasattr(frame, "tracks_3d") and hasattr(frame, "track_vis"):
            pts_cam = frame.tracks_3d 
            vis = frame.track_vis
            pts_cam_valid = pts_cam[vis]
            
            if len(pts_cam_valid) > 0:
                # Manually verify world alignment for viz
                pts_world = frame.pose.act(pts_cam_valid).cpu().numpy()
                rr.log("world/flow_points", rr.Points3D(
                    pts_world, colors=[255, 50, 50], radius=0.03
                ))

    print(f"Saved Rerun log to {args.output}")

if __name__ == "__main__":
    main()