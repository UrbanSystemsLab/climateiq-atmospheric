#!/bin/bash
#SBATCH --partition=normal # submit to the normal partition
#SBATCH --job-name=NYC_job
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log
#SBATCH --partition=normal 
#SBATCH --time=2-00:00:00 # Request 2 days of processing time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48 # Request 48 CPUs
#SBATCH --mem=192GB # Request 192 GB of memory


source /home/sislam27/.conda/envs/clim_data/lib/python3.10/venv/scripts/common/activate clim_data
python generate_UCPs.py NY
