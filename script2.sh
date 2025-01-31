#!/bin/bash

#SBATCH --job-name=maxFib      ## Name of the job
#SBATCH --output=maxFib.out    ## Output file
#SBATCH --time=10:00           ## Job Duration
#SBATCH --ntasks=2             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.

## Load the python interpreter
module purge


## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
srun --exclusive -n1 python script.py 90 &
srun --exclusive -n1 python script.py 100 &

wait