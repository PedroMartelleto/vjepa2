# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import torch
import torch.nn.functional as F

from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.pipeline.processors import ProcessedVideoStream, TrackAnythingProcessor
from vipe.streams.base import VideoStream, VideoFrame, FrameAttribute, StreamProcessor
from vipe.slam.interface import SLAMOutput

logger = logging.getLogger(__name__)

FrameAttribute.TRACKS_2D = "tracks_2d"
FrameAttribute.TRACKS_3D = "tracks_3d"
FrameAttribute.TRACK_VIS = "track_vis"


class CoTrackerProcessor(StreamProcessor):
    # ... (Same as before) ...
    def __init__(self, grid_size: int = 50, window_len: int = 30):
        super().__init__()
        self.grid_size = grid_size
        self.window_len = window_len
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda().eval()
        self.tracks_cache = {} 
        self.n_passes_required = 2
        self.step = 4

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.TRACKS_2D, FrameAttribute.TRACK_VIS}

    def _run_tracking(self, iterator):
        frames = list(iterator)
        if not frames: return
        
        # (1, T, 3, H, W)
        video_chunk = torch.stack([f.rgb for f in frames]).permute(0, 3, 1, 2)[None].cuda()
        
        self.model(video_chunk=video_chunk[:, :1], is_first_step=True, grid_size=self.grid_size)
        
        T = video_chunk.shape[1]
        for ind in range(0, T - self.step, self.step):
            pred_tracks, pred_vis = self.model(
                video_chunk=video_chunk[:, ind : ind + self.step * 2]
            ) 
            for t_local in range(self.step):
                t_global = ind + t_local
                if t_global < len(frames):
                    self.tracks_cache[frames[t_global].raw_frame_idx] = (
                        pred_tracks[0, t_local], 
                        pred_vis[0, t_local]     
                    )

    def update_iterator(self, previous_iterator, pass_idx):
        if pass_idx == 0:
            self._run_tracking(previous_iterator)
            yield from []
        else:
            for frame in previous_iterator:
                if frame.raw_frame_idx in self.tracks_cache:
                    t, v = self.tracks_cache[frame.raw_frame_idx]
                    setattr(frame, "tracks_2d", t)
                    setattr(frame, "track_vis", v)
                    frame.set_attribute(FrameAttribute.TRACKS_2D, t)
                    frame.set_attribute(FrameAttribute.TRACK_VIS, v)
                yield frame


class Lifter3DProcessor(StreamProcessor):
    """
    Lifts 2D tracks to 3D CAMERA coordinates using Metric Depth.
    Does NOT transform to World frame, preserving ego-motion in the flow.
    """
    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.TRACKS_3D}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        if not hasattr(frame, "tracks_2d") or frame.metric_depth is None or frame.intrinsics is None:
            return frame

        tracks = frame.tracks_2d
        H, W = frame.metric_depth.shape
        
        # 1. Sample depth at track locations
        u = tracks[:, 0]
        v = tracks[:, 1]
        
        # Normalize to [-1, 1]
        grid_x = (u / (W - 1)) * 2 - 1
        grid_y = (v / (H - 1)) * 2 - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)
        
        sampled_depth = F.grid_sample(
            frame.metric_depth.view(1, 1, H, W), 
            grid, 
            mode='nearest', 
            align_corners=True
        ).view(-1)
        
        # 2. Unproject to Camera 3D
        fx, fy, cx, cy = frame.intrinsics[:4]
        z = sampled_depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        pts_cam = torch.stack([x, y, z], dim=-1) # (N, 3)
        
        # CHANGED: We do NOT apply pose. 
        # Output is strictly relative to the camera center at this frame.
        
        setattr(frame, "tracks_3d", pts_cam)
        frame.set_attribute(FrameAttribute.TRACKS_3D, pts_cam)
        
        return frame


class DroidPreprocessingPipeline(DefaultAnnotationPipeline):
    # ... (Same as previous: load intrinsics, skip geocalib) ...
    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors = []
        dummy_frame = next(iter(video_stream))
        if dummy_frame.intrinsics is None:
            raise ValueError("DroidPreprocessingPipeline requires intrinsics in the stream!")
        
        if self.init_cfg.instance is not None:
             init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput) -> ProcessedVideoStream:
        processed_stream = super()._add_post_processors(view_idx, video_stream, slam_output)
        
        new_processors = processed_stream.processors + [
            CoTrackerProcessor(grid_size=50),
            Lifter3DProcessor()
        ]
        return ProcessedVideoStream(processed_stream.stream, new_processors)