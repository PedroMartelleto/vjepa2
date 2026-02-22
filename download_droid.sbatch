#!/bin/bash
#SBATCH --account=a144
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --job-name=download-droid
#SBATCH --output=download-droid.out
#SBATCH --error=download-droid.err

srun --environment=lyra bash -lc 'pip install gsutil && export PATH=$HOME/.local/bin:$PATH && gsutil -m rsync -r -x ".*stereo.*\.mp4$" "gs://gresearch/robotics/droid_raw" /iopsstor/scratch/cscs/pmartell/droid_raw/'