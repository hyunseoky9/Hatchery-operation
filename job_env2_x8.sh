#!/bin/bash

#SBATCH --job-name=env2_1POMDP8      ## Name of the job
#SBATCH --output=env2_1POMDP8.out    ## Output file
#SBATCH --time=20:59:00           ## Job Duration
#SBATCH --ntasks=20             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.

## not using the default python
module purge

## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 1 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 2 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 3 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 4 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 5 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 6 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 7 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 8 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 9 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 10 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 11 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 12 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 13 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 14 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 15 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 16 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 17 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 18 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 19 &
srun --exclusive -n1 python DRQN_env2_x_main_HPC.py 20 &

wait