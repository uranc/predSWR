#!/bin/bash
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=24GBS

source /gs/home/gascogalvezc/miniconda3/bin/activate
conda activate tfGPU215
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
srun  /gs/home/gascogalvezc/miniconda3/envs/tfGPU215/bin/python  /cs/projects/OWVinckSWR/DL/predSWR/pred_carmen.py --mode predict --model $1

echo "Job completed" > "/cs/projects/OWVinckSWR/DL/predSWR/summaries/bstatus/$2"

exit 0