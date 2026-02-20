# DROID Dataset Preprocessing Plan

## 1. Core Architecture
We adopt a "Hybrid Vipe Pipeline". We treat DROID intrinsics as trusted ground truth (hardware sensor data), but discard DROID extrinsics (unreliable robot odometry) in favor of Vipe's full bundle adjustment.

### Data Flow
1.  **Input**:
    *   **RGB**: Raw MP4 videos (loaded via `decord` directly to GPU).
    *   **Intrinsics**: `intrinsics.npy` (Trusted hardware calibration).
    *   **Poses**: Ignored (Re-estimated by Vipe).
2.  **Engine**: Vipe (GH200 Cluster).
3.  **Pipeline Steps**:
    *   **Init**: Load DROID Intrinsics. **Skip** `GeoCalib` to enforce metric scale from intrinsics.
    *   **SLAM**: Run Vipe Frontend + Backend BA to estimate **Poses**.
    *   **Depth**: `AdaptiveDepthProcessor` (Aligns DAv3 Giant to sparse SLAM points).
    *   **Motion**: `CoTracker3` (Online mode).
    *   **Lifting**: Lift 2D tracks to 3D using Metric Depth + Poses.
4.  **Output**: HuggingFace Parquet (Sharded).

## 2. Data Schema (HuggingFace / Parquet)
We store geometric data as **Binary Blobs (Numpy Bytes)** to ensure shape safety (avoiding flattened list confusion) and compression. RGB frames are **not** stored; only paths to source videos.

| Column | Data Type | Content |
| :--- | :--- | :--- |
| `video_uid` | `string` | Unique ID. |
| `video_path` | `string` | Relative path to MP4. |
| `pose` | `bytes` | `float32 [4, 4]` (Camera-to-World). |
| `intrinsics` | `bytes` | `float32 [4]` (`fx, fy, cx, cy`). |
| `depth_map` | `bytes` | `float16 [H, W]` (Metric Meters). |
| `tracks_2d` | `bytes` | `float16 [N, 2]` (Pixel coords). |
| `tracks_3d` | `bytes` | `float16 [N, 3]` (World coords). |
| `visibility` | `bytes` | `bool [N]` (Track visibility mask). |

## 3. Visualization & Sanity Check (Headless)
We use **Rerun (`.rrd`)** for headless 3D verification.
*   The script `check_droid_3d.py` runs the pipeline on one video and outputs a log file.
*   **Visual Checks**:
    *   Do 3D points form coherent structures?
    *   Do static objects stay static in 3D (no "swimming")?
    *   Does the camera frustum move correctly through the cloud?

## 4. Execution Strategy
1.  **Code**: Implement stream loader, custom pipeline, and visualizer.
2.  **Verify**: Run `check_droid_3d.py` on 1 clip, verify locally.
3.  **Scale**: Run `preprocess_droid.py` on the cluster.