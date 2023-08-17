#! /bin/bash

#SBATCH --job-name=test_simulation
#SBATCH --cluster=htc
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=devel
#SBATCH --array=1-6

cd $SCRATCH
module load Anaconda3/2022.05
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module load h5py/3.2.1-foss-2021a

rsync $DATA/main.py ./
config=$DATA/config.txt

n_neurons=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
max_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
n_paths=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

python -u main.py --n_neurons ${n_neurons} --max_time ${max_time} --n_paths ${n_paths}