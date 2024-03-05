#!/bin/bash
#SBATCH   --partition=normal            # submit   to the normal(default) partition
#SBATCH   --job-name=wrftest             # name the job
#SBATCH   --output=wrf.out        # write stdout/stderr   to named file
#SBATCH   --error=wrf.err      
#SBATCH   --time=7-00:00
#SBATCH   --nodes=1
#SBATCH   --ntasks=48
#SBATCH   --cpus-per-task=1
#SBATCH   --mem-per-cpu=2GB
#SBATCH   --constraint=intel

# run real.exe
srun -n 16 ./real.exe >& real.log

# run wrf
srun -n 48 ./wrf.exe >& wrf.log
