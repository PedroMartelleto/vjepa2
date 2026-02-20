import decord
import torch
import numpy as np
from pathlib import Path
from vipe.streams.base import VideoStream, VideoFrame, CameraType

class DecordDroidStream(VideoStream):
    def __init__(self, video_path: Path, intrinsics_path: Path, ctx_device: str = "cuda:0"):
        super().__init__()
        self.path = video_path
        self._name = video_path.stem
        
        # Load Intrinsics (N, 4) -> [fx, fy, cx, cy]
        # DROID intrinsics are usually roughly constant, but provided per frame.
        self.intrinsics_np = np.load(intrinsics_path)
        
        # Init Decord
        if "cuda" in ctx_device:
            device_id = int(ctx_device.split(":")[-1])
            ctx = decord.gpu(device_id)
        else:
            ctx = decord.cpu(0)
            
        self.vr = decord.VideoReader(str(video_path), ctx=ctx)
        self._fps = self.vr.get_avg_fps()
        self._len = len(self.vr)
        
        # Validation
        if len(self.intrinsics_np) != self._len:
            # Handle mismatch (sometimes DROID has +/- 1 frame)
            min_len = min(len(self.intrinsics_np), self._len)
            self._len = min_len
            self.intrinsics_np = self.intrinsics_np[:min_len]
        
        h, w, _ = self.vr[0].shape
        self._size = (h, w)
        self.device = torch.device(ctx_device)

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
        
        # Load intrinsic for this frame
        # Shape: (4,) [fx, fy, cx, cy]
        intr = torch.from_numpy(self.intrinsics_np[self.idx]).float().to(self.device)
            
        return VideoFrame(
            raw_frame_idx=self.idx,
            rgb=rgb,
            intrinsics=intr,
            camera_type=CameraType.PINHOLE # DROID is pinhole
        )