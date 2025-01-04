#!/bin/bash
   
#SBATCH --job-name=tensorflow_gpu
#SBATCH --partition=GPUshort,GPUlong,GPUtest
#SBATCH --gpus=rtxa6000:1  # Request single GPU
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=7800
#SBATCH --time=0-4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de

source /mnt/hpc/slurm/uranc/anacond/bin/activate
conda activate tf_2_14
export HDF5_USE_FILE_LOCKING=FALSE 
export CUDA_VISIBLE_DEVICES=0

# Run parallel jobs on single GPU with memory limits
for i in {0..3}; do
    export TF_MEMORY_ALLOCATION="11264" # Divide GPU memory by 4 (12GB/4 = 3GB per process)
    srun --exclusive --cpus-per-task=2 --mem-per-cpu=7800 --exact \
        /mnt/hpc/slurm/uranc/anacond/envs/tf_2_14/bin/python \
        /cs/projects/MWNaturalPredict/DL/predSWR/pred.py \
        --mode tune_worker --model $1 \
        > "log_job_${i}.txt" 2>&1 &
done

wait
exit 0
