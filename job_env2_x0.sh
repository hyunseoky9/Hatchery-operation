#!/bin/bash

#SBATCH --job-name=env2_1POMDP0      ## Name of the job
#SBATCH --output=env2_1POMDP0.out    ## Output file
#SBATCH --time=05:30:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.

## not using the default python
module purge

## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 1 &

wait