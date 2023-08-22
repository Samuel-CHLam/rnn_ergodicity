#! /bin/bash

#SBATCH --job-name=test_simulation
#SBATCH --cluster=htc
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --array=6

cd $SCRATCH
module load Anaconda3/2022.05
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module load tqdm/4.61.2-GCCcore-10.3.0

rsync $DATA/main.py ./
rsync $DATA/simulation.py ./
config=$DATA/config.txt

n_neurons=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
max_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
n_paths=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

python -u main.py --n_neurons ${n_neurons} --max_time ${max_time} --n_paths ${n_paths} --no_rolling_mean --return_summary=True --discard_history > output.out