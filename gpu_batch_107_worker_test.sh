#!/bin/bash
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=GPUlongx86
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=7800
#SBATCH --time=00-10:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de
#SBATCH --gres-flags=enforce-binding

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=1 
export TF_CUDNN_USE_AUTOTUNE=0
export LD_PRELOAD=/usr/lib64/libcuda.so.1   # or whatever ldconfig shows

srun  rm -rf ~/.nv/ComputeCache ~/.nv/GLCache ~/.cache/torch_extensions 2>/dev/null || true
srun /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/pred.py --mode tune_worker --tag $1

exit 0
