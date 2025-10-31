#!/bin/bash
   
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=VINCKGPU
#SBATCH --gpus=rtx2080:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=5800
#SBATCH --time=2-23:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
export CUDA_VISIBLE_DEVICES=0


srun /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/pred.py --mode tune_worker --tag $1

exit 0
