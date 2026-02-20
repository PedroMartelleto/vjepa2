# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import io
import json
import numpy as np
import datasets
import torch
import hydra
from pathlib import Path
from tqdm import tqdm

from vipe.streams.droid import DecordDroidStream
from vipe.pipeline.droid import DroidPreprocessingPipeline

def to_bytes(tensor):
    if tensor is None: return b""
    f = io.BytesIO()
    if tensor.dtype == torch.float32 and tensor.numel() > 100: 
        tensor = tensor.half()
    np.save(f, tensor.cpu().numpy())
    return f.getvalue()

def parse_droid_intrinsics(json_path: Path, camera_serial: str):
    """
    Parses the DROID metadata.json to find intrinsics for a specific camera serial.
    Adapts to common DROID/R2D2 schema variations.
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    # 1. Try "cameras" list approach
    cameras = meta.get("cameras", [])
    if isinstance(cameras, list):
        for cam in cameras:
            # Check for serial string or int match
            if str(cam.get("serial_number", "")) == camera_serial or str(cam.get("serial", "")) == camera_serial:
                # Look for 'intrinsics' or 'K'
                if "intrinsics" in cam: 
                    return np.array(cam["intrinsics"]) # [fx, fy, cx, cy]
                if "K" in cam:
                    K = np.array(cam["K"]) # [3, 3] or flat
                    if K.shape == (3,3):
                        return np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
    
    # 2. Try dict approach "serial" -> data
    cameras_dict = meta.get("cameras", {})
    if isinstance(cameras_dict, dict):
        if camera_serial in cameras_dict:
            cam = cameras_dict[camera_serial]
            if "intrinsics" in cam: return np.array(cam["intrinsics"])

    # 3. Fallback: Check top level calibration (sometimes single camera)
    if "intrinsics" in meta:
        return np.array(meta["intrinsics"])

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Root of DROID dataset")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output HF dataset dir")
    parser.add_argument("--shard_size", type=str, default="2GB", help="Max shard size")
    args = parser.parse_args()
    
    def generator():
        overrides = [
            "pipeline.instance=vipe.pipeline.droid.DroidPreprocessingPipeline",
            "pipeline.slam.optimize_intrinsics=false", 
            "pipeline.output.save_artifacts=false",
            "pipeline.output.save_viz=false"
        ]
        
        with hydra.initialize_config_dir(config_dir="../configs", version_base=None):
            cfg = hydra.compose("default", overrides=overrides)

        pipeline = DroidPreprocessingPipeline(**cfg.pipeline)
        
        # 1. Find all Episodes (identified by metadata json)
        # We look for files starting with 'metadata_'
        metadata_files = sorted(args.input_dir.rglob("metadata_*.json"))
        print(f"Found {len(metadata_files)} episodes (metadata files).")
        
        for meta_path in tqdm(metadata_files, desc="Processing Episodes"):
            episode_dir = meta_path.parent
            recordings_dir = episode_dir / "recordings" / "MP4"
            
            if not recordings_dir.exists():
                # DROID directory structure sometimes varies
                # Try finding MP4s recursively in this episode folder
                mp4s = list(episode_dir.rglob("*.mp4"))
            else:
                mp4s = list(recordings_dir.glob("*.mp4"))

            if not mp4s:
                continue

            for vid_path in mp4s:
                # Camera serial is filename without extension (e.g., '13263313')
                cam_serial = vid_path.stem
                
                # Extract Intrinsics
                intrinsics = parse_droid_intrinsics(meta_path, cam_serial)
                
                if intrinsics is None:
                    print(f"Skipping {vid_path}: Could not find intrinsics for serial {cam_serial} in {meta_path.name}")
                    continue
                
                try:
                    # Init stream with explicit intrinsics
                    stream = DecordDroidStream(vid_path, intrinsics)
                    
                    pipeline.return_output_streams = True
                    output = pipeline.run(stream)
                    final_stream = output.output_streams[0]
                    
                    for frame in final_stream:
                        yield {
                            "video_uid": vid_path.stem,
                            "video_path": str(vid_path.relative_to(args.input_dir)),
                            "frame_idx": frame.raw_frame_idx,
                            "pose": to_bytes(frame.pose.matrix()),
                            "intrinsics": to_bytes(frame.intrinsics),
                            "depth_map": to_bytes(frame.metric_depth),
                            "tracks_2d": to_bytes(frame.tracks_2d) if hasattr(frame, 'tracks_2d') else b"",
                            "tracks_3d": to_bytes(frame.tracks_3d) if hasattr(frame, 'tracks_3d') else b"",
                            "visibility": to_bytes(frame.track_vis) if hasattr(frame, 'track_vis') else b""
                        }
                    
                    del stream, output, final_stream
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {vid_path}: {e}")
                    continue

    features = datasets.Features({
        "video_uid": datasets.Value("string"),
        "video_path": datasets.Value("string"),
        "frame_idx": datasets.Value("int32"),
        "pose": datasets.Value("binary"),
        "intrinsics": datasets.Value("binary"),
        "depth_map": datasets.Value("binary"),
        "tracks_2d": datasets.Value("binary"),
        "tracks_3d": datasets.Value("binary"),
        "visibility": datasets.Value("binary"),
    })

    print("Starting dataset generation...")
    ds = datasets.Dataset.from_generator(generator, features=features, num_proc=1)
    ds.save_to_disk(str(args.output_dir), max_shard_size=args.shard_size)
    print("Done!")

if __name__ == "__main__":
    main()