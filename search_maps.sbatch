#!/bin/bash

#SBATCH --job-name=search
#SBATCH --output=search_%j.log
#SBATCH --error=search_%j.err 
#SBATCH --time=2-00:00:00     
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:volta:1

module load anaconda/Python-ML-2025a
python /home/gridsan/jroney/broeyjoey/search_maps.py
