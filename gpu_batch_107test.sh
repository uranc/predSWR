#!/bin/bash
   
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=GPUshort,GPUlong,GPUtest
#SBATCH --gpus=rtxa6000:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=7800
#SBATCH --time=0-4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14

export HDF5_USE_FILE_LOCKING=FALSE 
export CUDA_VISIBLE_DEVICES=0

# GPU configuration - simplified to avoid conflicts
export CUDA_VISIBLE_DEVICES=0
export TF_MEMORY_ALLOCATION=11264

# Remove TF_FORCE_GPU_ALLOW_GROWTH since we're handling memory in code
# Rest of GPU/CUDA settings remain the same

# Add memory allocation limit (in MB)

# Optimize TensorFlow performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_USE_CUDNN=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=1

# Optional: Set logging level
export TF_CPP_MIN_LOG_LEVEL=2  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR


srun /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/experiments/$1/pred.py --mode train --model $1

exit 0
