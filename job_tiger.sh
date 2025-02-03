#!/bin/bash

#SBATCH --job-name=tigerPOMDP1      ## Name of the job
#SBATCH --output=tigerPOMDP1.out    ## Output file
#SBATCH --time=08:40:00           ## Job Duration
#SBATCH --ntasks=10             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=8      ## The number of threads the code will use
#SBATCH --mem-per-cpu=2G     ## Real memory(MB) per CPU required by the job.

## not using the default python
module purge


## Execute the python script and pass the argument/input '90'
source ~/miniconda3/bin/activate py310
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 1 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 2 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 3 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 4 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 5 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 6 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 7 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 8 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 9 &
srun --exclusive -n1 python DRQN_tiger_main_HPC.py 10 &

wait