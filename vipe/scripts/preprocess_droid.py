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
from huggingface_hub import hf_hub_download

from vipe.streams.droid import DecordDroidStream
from vipe.pipeline.droid import DroidPreprocessingPipeline

def to_bytes(tensor):
    if tensor is None: return b""
    if isinstance(tensor, np.ndarray):
        f = io.BytesIO()
        np.save(f, tensor)
        return f.getvalue()
    if tensor.dtype == torch.float32 and tensor.numel() > 100: 
        tensor = tensor.half()
    f = io.BytesIO()
    np.save(f, tensor.cpu().numpy())
    return f.getvalue()

def load_droid_intrinsics_db(cache_dir: Path | None = None):
    """Downloads and loads the KarlP/droid intrinsics.json"""
    print("Downloading intrinsics.json from KarlP/droid...")
    json_path = hf_hub_download(
        repo_id="KarlP/droid",
        filename="intrinsics.json",
        repo_type="dataset",
        cache_dir=cache_dir
    )
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_intrinsics(db, episode_id, serial):
    """
    Look up intrinsics.
    KarlP DB Format: {episode_id: {serial: [fx, cx, fy, cy]}}
    Vipe Expected: [fx, fy, cx, cy]
    """
    if episode_id not in db:
        return None
    
    ep_data = db[episode_id]
    if serial not in ep_data:
        return None
        
    # [fx, cx, fy, cy]
    vals = ep_data[serial]
    
    # Swap to [fx, fy, cx, cy]
    return np.array([vals[0], vals[2], vals[1], vals[3]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Root of DROID dataset")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output HF dataset dir")
    parser.add_argument("--shard_size", type=str, default="2GB", help="Max shard size")
    args = parser.parse_args()
    
    # Load Intrinsics DB
    intrinsics_db = load_droid_intrinsics_db(cache_dir=args.input_dir / ".cache")
    print(f"Loaded intrinsics for {len(intrinsics_db)} episodes.")

    def generator():
        overrides = [
            "pipeline.instance=vipe.pipeline.droid.DroidPreprocessingPipeline",
            "pipeline.slam.optimize_intrinsics=false", # TRUST GT INTRINSICS
            "pipeline.output.save_artifacts=false",
            "pipeline.output.save_viz=false"
        ]
        
        with hydra.initialize_config_dir(config_dir="../configs", version_base=None):
            cfg = hydra.compose("default", overrides=overrides)

        pipeline = DroidPreprocessingPipeline(**cfg.pipeline)
        
        # Find Episodes via metadata_*.json
        metadata_files = sorted(args.input_dir.rglob("metadata_*.json"))
        print(f"Found {len(metadata_files)} metadata files.")
        
        for meta_path in tqdm(metadata_files, desc="Processing Episodes"):
            # Extract Episode ID from filename: metadata_UUID.json -> UUID
            episode_id = meta_path.stem.replace("metadata_", "")
            
            episode_dir = meta_path.parent
            mp4_files = sorted(list(episode_dir.rglob("*.mp4")))
            
            for vid_path in mp4_files:
                serial = vid_path.stem
                
                # Lookup Intrinsics
                intr = get_intrinsics(intrinsics_db, episode_id, serial)
                
                if intr is None:
                    # Try fallback: sometimes episode_id in DB differs slightly or has/missing timestamp
                    # But KarlP doc says it matches metadata filename suffix.
                    # Verify if user has matching keys manually if this fails often.
                    pass 

                if intr is None:
                    # Hard fail or skip? Skip for now.
                    # print(f"Skipping {vid_path}: No intrinsics in DB for {episode_id}/{serial}")
                    continue
                
                try:
                    # Init Stream with GT Intrinsics
                    stream = DecordDroidStream(
                        vid_path, 
                        intrinsics=intr, 
                        # h5_path=... (Optional, skip poses for speed as we trust Vipe SLAM)
                    )
                    
                    pipeline.return_output_streams = True
                    output = pipeline.run(stream)
                    final_stream = output.output_streams[0]
                    
                    for frame in final_stream:
                        yield {
                            "video_uid": episode_id,
                            "video_path": str(vid_path.relative_to(args.input_dir)),
                            "camera_serial": serial,
                            "frame_idx": frame.raw_frame_idx,
                            "pose": to_bytes(frame.pose.matrix()),
                            "intrinsics": to_bytes(frame.intrinsics),
                            "depth_map": to_bytes(frame.metric_depth),
                            "tracks_2d": to_bytes(frame.tracks_2d) if hasattr(frame, 'tracks_2d') else b"",
                            "tracks_3d": to_bytes(frame.tracks_3d) if hasattr(frame, 'tracks_3d') else b"",
                            "visibility": to_bytes(frame.track_vis) if hasattr(frame, 'track_vis') else b"",
                        }
                    
                    del stream, output, final_stream
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {vid_path}: {e}")
                    continue

    features = datasets.Features({
        "video_uid": datasets.Value("string"),
        "video_path": datasets.Value("string"),
        "camera_serial": datasets.Value("string"),
        "frame_idx": datasets.Value("int32"),
        "pose": datasets.Value("binary"),
        "intrinsics": datasets.Value("binary"),
        "depth_map": datasets.Value("binary"),
        "tracks_2d": datasets.Value("binary"),
        "tracks_3d": datasets.Value("binary"),
        "visibility": datasets.Value("binary"),
    })

    print("Generating dataset...")
    ds = datasets.Dataset.from_generator(generator, features=features, num_proc=1)
    ds.save_to_disk(str(args.output_dir), max_shard_size=args.shard_size)
    print("Done!")

if __name__ == "__main__":
    main()