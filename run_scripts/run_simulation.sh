#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "cpm"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

eval "$(conda shell.bash hook)"
source activate synmorph

python run_param_scan.py ${SLURM_ARRAY_TASK_ID}




#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}