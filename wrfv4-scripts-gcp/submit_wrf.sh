#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=6
spack env activate wrfv4-c2

# number of threads to use when creating parallel regions
export OMP_NUM_THREADS=1

scontrol show hostnames > hostfile

mpirun -n 120 -ppn 20 -hostfile hostfile wrf.exe
