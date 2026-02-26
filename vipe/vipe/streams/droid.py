# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import decord
import h5py
import numpy as np
import torch

from vipe.streams.base import CameraType, VideoFrame, VideoStream


# CRITICAL: Configure Decord to output PyTorch Tensors directly.
# This prevents 'NDArray object has no attribute cpu' errors.
decord.bridge.set_bridge("torch")


class DecordDroidStream(VideoStream):
    """
    Stream for DROID dataset.
    - Loads video via Decord (Direct to PyTorch Tensor).
    - Accepts explicit Intrinsics vector [fx, fy, cx, cy].
    - Optionally loads original DROID poses from H5 for archival.
    """

    def __init__(
        self,
        video_path: Path,
        intrinsics: np.ndarray,
        h5_path: Path | None = None,
        camera_serial: str | None = None,
        ctx_device: str = "cuda:0",
    ):
        super().__init__()
        self.path = video_path
        self._name = video_path.stem
        self.droid_poses = None

        # Intrinsics: Expecting [fx, fy, cx, cy]
        self.intrinsics_vec = torch.from_numpy(intrinsics).float()

        # Init Decord Context
        if "cuda" in ctx_device:
            try:
                device_id = int(ctx_device.split(":")[-1])
                ctx = decord.gpu(device_id)
            except Exception:
                print(f"Decord GPU init failed for {ctx_device}, falling back to CPU")
                ctx = decord.cpu(0)
        else:
            ctx = decord.cpu(0)

        self.vr = decord.VideoReader(str(video_path), ctx=ctx)
        self._fps = self.vr.get_avg_fps()
        self._len = len(self.vr)
        self._size = (self.vr[0].shape[0], self.vr[0].shape[1])

        self.device = torch.device(ctx_device)
        self.intrinsics_vec = self.intrinsics_vec.to(self.device)

        # Attempt to load DROID Extrinsics from H5 if provided
        if h5_path and h5_path.exists() and camera_serial:
            try:
                with h5py.File(h5_path, "r") as f:
                    extrinsics_grp = f["observation/camera_extrinsics"]
                    key_match = None
                    for k in extrinsics_grp.keys():
                        if str(camera_serial) in k:
                            key_match = k
                            break

                    if key_match:
                        self.droid_poses = np.array(extrinsics_grp[key_match])
                        if len(self.droid_poses) != self._len:
                            min_len = min(len(self.droid_poses), self._len)
                            self.droid_poses = self.droid_poses[:min_len]
            except Exception:
                pass

    def frame_size(self) -> tuple[int, int]:
        return self._size

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self) -> VideoFrame:
        self.idx += 1
        if self.idx >= self._len:
            raise StopIteration

        # Thanks to set_bridge("torch"), this returns a torch.Tensor
        rgb = self.vr[self.idx]

        # Ensure correct formatting (Float 0-1)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        # Ensure it's on the correct device (Decord might return CPU tensor if context failed)
        if rgb.device != self.device:
            rgb = rgb.to(self.device)

        frame = VideoFrame(raw_frame_idx=self.idx, rgb=rgb, intrinsics=self.intrinsics_vec, camera_type=CameraType.PINHOLE)

        if self.droid_poses is not None and self.idx < len(self.droid_poses):
            setattr(frame, "droid_pose_raw", self.droid_poses[self.idx])

        return frame
