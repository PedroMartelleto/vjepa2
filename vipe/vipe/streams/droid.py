# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import decord
import torch
import numpy as np
from pathlib import Path
from vipe.streams.base import VideoStream, VideoFrame, CameraType

class DecordDroidStream(VideoStream):
    """
    Loads video from DROID dataset using Decord.
    Accepts explicit intrinsics (vector).
    """
    def __init__(self, 
                 video_path: Path, 
                 intrinsics: np.ndarray, 
                 ctx_device: str = "cuda:0"):
        super().__init__()
        self.path = video_path
        self._name = video_path.stem
        
        # Intrinsics: Expecting [fx, fy, cx, cy]
        self.intrinsics_vec = torch.from_numpy(intrinsics).float()
        
        # Init Decord
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
            
        rgb = self.vr[self.idx]
        if isinstance(rgb, decord.NDArray):
            rgb = torch.from_dlpack(rgb)
            
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        
        return VideoFrame(
            raw_frame_idx=self.idx,
            rgb=rgb,
            intrinsics=self.intrinsics_vec, # Constant per video
            camera_type=CameraType.PINHOLE
        )