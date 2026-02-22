# DROID Dataset Preprocessing Plan

## 1. Core Architecture
We implement a **Hybrid Pipeline**.
*   **Intrinsics**: Trusted Ground Truth (extracted from `.svo` recordings).
*   **Poses**: Re-estimated by Vipe (Full Bundle Adjustment). DROID extrinsics are unreliable and discarded (except for archival).
*   **Depth**: Estimated by Vipe (DAv3 Giant) + Metric Alignment to the new Vipe Poses.
*   **Motion**: CoTracker3 (Online).

### Data Flow
1.  **Input Structure**:
    *   `trajectory.h5`: Metadata & original poses.
    *   `recordings/SVO/*.svo`: Source of **GT Intrinsics**.
    *   `recordings/MP4/*.mp4`: Video Source (Decord).
2.  **Engine**: Vipe (GH200 Cluster).
3.  **Pipeline Steps**:
    *   **Init**: Inject GT Intrinsics. **Skip** `GeoCalib`.
    *   **SLAM**: Vipe Backend estimates Poses (Fixing intrinsics).
    *   **Depth**: `AdaptiveDepthProcessor` (Aligns DAv3 to Vipe SLAM points).
    *   **Motion**: `CoTracker3` (Online).
    *   **Lifting**: Lift 2D tracks to 3D Camera Frame.
4.  **Output**: HuggingFace Parquet (Binary Blobs).

## 2. Execution Strategy
1.  **Loader**: `preprocess_droid.py` iterates episodes.
2.  **SVO Read**: For each episode, locate `.svo` files, instantiate `SVOReader`, read intrinsics matrix.
3.  **Stream**: Pass the matrix vector `[fx, fy, cx, cy]` to `DecordDroidStream`.
4.  **Process**: Run Vipe.

## 3. Requirements
*   `pyzed` or DROID `SVOReader` must be available in the python path to read SVOs.

## 4. Visualization & Sanity Check (Headless/SSH)

Since we cannot open a GUI, we will use **Rerun (`.rrd` exports)**.

### The Visualization Pipeline (`pipeline/check_droid.py`)
This script will process **one** sequence and generate a `debug.rrd` file containing:

1.  **RGB Video**: Background layer.
2.  **Frustums**: Visualizing the DROID camera trajectory.
3.  **Depth Cloud**: The dense depth map unprojected to 3D (colored by RGB). *Check: Is the scale consistent with the frustum size?*
4.  **2D Tracks**: Lines drawn on the image plane.
5.  **3D Flows (New)**:
    *   Draw the 3D trajectories of the tracked points in the world space.
    *   *Visual Check*: Static objects should produce 3D points that stay stationary (clumps). Moving objects should produce 3D streaks (worms).
6.  **Error Heatmap**: Color-code points by their reprojection error. Red points = Inconsistent geometry (likely dynamic objects or bad depth).

## 5. Execution Steps

### Step 1: Verification (Local/Dev)
1.  Implement `DecordDroidStream`.
2.  Run `check_droid.py` on 1 sequence.
3.  Download `debug.rrd` and view locally to confirm coordinates and alignment.

### Step 2: Processor Implementation
1.  Implement `CoTrackerProcessor` (wrapping `torch.hub`).
2.  Modify `AdaptiveDepthProcessor` to accept "external poses" instead of waiting for SLAM.
3.  Implement `LiftingProcessor` to compute 3D flows from Tracks + Depth + Poses.

### Step 3: Production Run (Cluster)
1.  Script `preprocess_to_hf.py` initializes the HF `ShardWriter`.
2.  Launch via `torchrun` or `ray` across GH200 nodes.
3.  Workers read MP4s $\to$ Compute Features $\to$ Write Binary Blobs to Parquet.
4.  Final step: Push to HuggingFace Hub or consolidate shards.