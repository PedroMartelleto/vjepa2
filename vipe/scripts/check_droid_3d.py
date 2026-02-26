# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import json

from pathlib import Path

import hydra
import numpy as np
import rerun as rr

from huggingface_hub import hf_hub_download

from vipe.pipeline.droid import DroidPreprocessingPipeline
from vipe.streams.droid import DecordDroidStream


def load_intrinsics_db():
    print("Fetching intrinsics.json...")
    path = hf_hub_download(repo_id="KarlP/droid", filename="intrinsics.json", repo_type="model")
    with open(path, "r") as f:
        return json.load(f)


def find_metadata_file(video_path: Path):
    curr = video_path.parent
    for _ in range(4):
        metas = list(curr.glob("metadata_*.json"))
        if metas:
            return metas[0]
        curr = curr.parent
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path, help="Path to .mp4 file")
    parser.add_argument("--output", type=Path, default="debug_droid.rrd")
    args = parser.parse_args()

    # 1. Identify Episode ID
    meta_path = find_metadata_file(args.video_path)
    if not meta_path:
        print("Could not find metadata_*.json parent file.")
        return

    episode_id = meta_path.stem.replace("metadata_", "")
    serial = args.video_path.stem
    print(f"Episode: {episode_id}, Camera: {serial}")

    # 2. Get Intrinsics
    db = load_intrinsics_db()
    if episode_id not in db or serial not in db[episode_id]:
        print(f"Intrinsics not found in KarlP/droid for {episode_id}/{serial}")
        return

    vals = db[episode_id][serial]["cameraMatrix"]  # [fx, cx, fy, cy]
    intrinsics = np.array([vals[0], vals[2], vals[1], vals[3]])
    print(f"Intrinsics: {intrinsics}")

    # 3. Setup Rerun & Pipeline
    rr.init("droid_sanity_check", spawn=False)
    rr.save(str(args.output))

    overrides = [
        "pipeline=default",
        "pipeline.instance=vipe.pipeline.droid.DroidPreprocessingPipeline",
        "pipeline.slam.keyframe_depth=unidepth-l",
        "pipeline.post.depth_align_model=adaptive_unidepth-l_svda",
        "pipeline.slam.optimize_intrinsics=false",  # TRUST GT
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_viz=false",
        "pipeline.init.instance=null",
    ]

    config_dir = Path.cwd() / "configs"

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose("default", overrides=overrides)

    stream = DecordDroidStream(args.video_path, intrinsics=intrinsics)
    pipeline = DroidPreprocessingPipeline(
        init=cfg.pipeline.init, slam=cfg.pipeline.slam, post=cfg.pipeline.post, output=cfg.pipeline.output
    )

    pipeline.return_output_streams = True
    print("Running Pipeline...")
    try:
        output = pipeline.run(stream)
        final_stream = output.output_streams[0]
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        return

    # 4. Log
    print("Logging to Rerun...")
    for frame in final_stream:
        rr.set_time_sequence("frame", frame.raw_frame_idx)
        rr.log("camera/image", rr.Image(frame.rgb.cpu().numpy()))

        if frame.pose is not None:
            t = frame.pose.translation().cpu().numpy()
            q = frame.pose.quaternion().cpu().numpy()
            rr.log("camera", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q), from_parent=False))

        if hasattr(frame, "tracks_3d") and hasattr(frame, "track_vis"):
            pts_cam = frame.tracks_3d
            vis = frame.track_vis
            pts_valid = pts_cam[vis]
            if len(pts_valid) > 0 and frame.pose is not None:
                # Manual World Transform for Vis
                pts_world = frame.pose.act(pts_valid).cpu().numpy()
                rr.log("world/flow_points", rr.Points3D(pts_world, colors=[255, 50, 50], radius=0.03))

    print(f"Saved log to {args.output}")


if __name__ == "__main__":
    main()
