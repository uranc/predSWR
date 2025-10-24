#!/bin/bash
#SBATCH --job-name=tensorflow_gpu_power
#SBATCH --partition=GPUlongppc
#SBATCH --gpus=v100:1
#SBATCH --constraint=cpu_power9
#SBATCH --cpus-per-gpu=30
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cem.uran@esi-frankfurt.de


source /mnt/hpc/slurm/uranc/pcond/bin/activate
conda activate openTF_2_14

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export HDF5_USE_FILE_LOCKING=FALSE 
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/mnt/hpc/slurm/uranc/pcond/envs/openTF_2_14
export CUDA_DIR=/mnt/hpc/slurm/uranc/pcond/envs/openTF_2_14


# Tell XLA where CUDA toolkit (in your env) lives
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_PREFIX}"

# Ensure your env’s tools/libs are used
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Sanity checks (fail fast)
which ptxas && ptxas --version
(ls ${CONDA_PREFIX}/nvvm/libdevice/libdevice*.bc || ls ${CONDA_PREFIX}/targets/x86_64-linux/libdevice/libdevice*.bc)
python -c "import os; print('XLA_FLAGS=', os.environ.get('XLA_FLAGS'))"

srun /mnt/hpc/slurm/uranc/pcond/envs/openTF_2_14/bin/python /mnt/hpc/projects/MWNaturalPredict/DL/predSWR/pred.py --mode tune_worker --tag $1

exit 0


