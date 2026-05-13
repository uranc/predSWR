#!/bin/bash
#SBATCH --job-name=blackwell_tune
#SBATCH --partition=GPUblackwell
#SBATCH --gres=mps:4
#SBATCH --time=01-23:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de
#SBATCH --gres-flags=enforce-binding

# 1. Activate the pure conda-forge environment
source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_blackwell_cf

# 2. Set safe, necessary environment variables
export TF_USE_LEGACY_KERAS=1
export HDF5_USE_FILE_LOCKING=FALSE 
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 3. Clean up old caches to prevent cuDNN 8 ghosts from interfering
srun rm -rf ~/.nv/ComputeCache ~/.nv/GLCache ~/.cache/torch_extensions 2>/dev/null || true

# 4. Launch the Optuna worker
srun /mnt/hpc/slurm/uranc/anacond/envs/tf_blackwell_cf/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/pred.py --mode tune_worker --tag $1

exit 0