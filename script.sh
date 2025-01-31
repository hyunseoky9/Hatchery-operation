#!/bin/bash

#SBATCH --job-name=maxFib      ## Name of the job
#SBATCH --output=maxFib.out    ## Output file
#SBATCH --time=10:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=1      ## The number of threads the code will use
#SBATCH --mem-per-cpu=100M     ## Real memory(MB) per CPU required by the job.

## Load the python interpreter
module load python

## Execute the python script and pass the argument/input '90'
srun python script.py 90