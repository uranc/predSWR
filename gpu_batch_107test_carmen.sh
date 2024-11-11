#!/bin/bash
   
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=VINCKGPU
#SBATCH --gpus=rtxa6000:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=7000
#SBATCH --time=0-4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=carmen.gascogalvez@ru.nl

source /gs/home/gascogalvezc/miniconda3/bin/activate
conda activate tfGPU215
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 

srun /gs/home/gascogalvezc/miniconda3/envs/tfGPU215/bin/python /cs/projects/OWVinckSWR/DL/predSWR/experiments_carmen/$1/pred_carmen.py --mode train --model $1

exit 0