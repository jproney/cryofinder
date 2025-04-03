#!/bin/bash

#SBATCH --job-name=resnet
#SBATCH --output=resnet_%j.log
#SBATCH --error=resnet_%j.err 
#SBATCH --time=2-00:00:00     
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=8             
#SBATCH --gres=gpu:volta:1

module load anaconda/Python-ML-2025a
python /home/gridsan/jroney/broeyjoey/train.py --exp_name resnet_cluster_v1 --batch_size 32
