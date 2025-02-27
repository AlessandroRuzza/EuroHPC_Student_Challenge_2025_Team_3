#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:46:40

module load mpi4py
module load SciPy-bundle

srun --output=$2/${1:0:-4}.stdout python MPI/mpi.py ../instances/$1 $2 --cpusPerTask 32
