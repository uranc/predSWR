#!/bin/bash
   
#SBATCH --job-name=tensorflow2
#SBATCH --partition=VINCKGPU
#SBATCH --gpus=rtx8000:1
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-cpu=6000
#SBATCH --time=0-08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/anacond/bin/activate
# conda activate powertf
conda activate tfGPU214
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
srun /mnt/hpc/slurm/uranc/anacond/envs/tfGPU215/bin/python /mnt/hpc/projects/MWNaturalPredict/DL/pconv_fft_v2_tf2/experiments/$1/pred.py --mode train --model $1 

exit 0

