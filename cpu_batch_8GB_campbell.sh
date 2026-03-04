#!/bin/bash
#SBATCH --job-name=SWR_pred
#SBATCH --partition=8GB

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 

# $1=model, $2=val_id, $3=start_ind, $4=tag
srun /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python /cs/projects/MWNaturalPredict/DL/predSWR/predAllen.py \
    --model "$1" \
    --val "$2" \
    --start_ind "$3" \
    --tag "$4"

exit 0