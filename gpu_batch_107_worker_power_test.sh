#!/bin/bash
#SBATCH --job-name=tensorflow_gpu_power
#SBATCH --partition=GPUlongppc
#SBATCH --gpus=v100:1
#SBATCH --constraint=cpu_power9
#SBATCH --cpus-per-gpu=30
#SBATCH --time=0-03:59:59
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/pcond/bin/activate
conda activate openTF_2_14

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
export CUDA_VISIBLE_DEVICES=0

srun /mnt/hpc/slurm/uranc/pcond/envs/openTF/bin/python /mnt/hpc/projects/MWNaturalPredict/DL/predSWR/pred.py --mode tune_worker --tag $1

exit 0

