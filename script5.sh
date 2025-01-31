#!/bin/bash

#SBATCH --job-name=tigerPOMDP5     ## Name of the job
#SBATCH --output=tigerPOMDP5.out    ## Output file
#SBATCH --time=00:40:00           ## Job Duration
#SBATCH --ntasks=3             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.

## Load the python interpreter
module purge


## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 1 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 2 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 3 &

wait