#!/bin/bash

#SBATCH --job-name=project
#SBATCH --output=project_%j.log
#SBATCH --error=project_%j.err 
#SBATCH --time=01:00:00     
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=1             
#SBATCH --partition=sched_cdrennan
#SBATCH --gres=gpu:1

module load anaconda/2020.11
conda activate cryodrgn
cd /nobackup/users/jamesron/broeyjoey
python batch_project.py /nobackup/users/jamesron/siren_maps.csv /nobackup/users/jamesron/raw_maps_07072023/ /nobackup/users/jamesron/projections --healpy-grid 2 --start $1 --end $2
