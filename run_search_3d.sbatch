#!/bin/bash

#SBATCH --job-name=search
#SBATCH --output=search_%j.log
#SBATCH --error=search_%j.err 
#SBATCH --time=2-00:00:00     
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=128G

module load anaconda/Python-ML-2025a
python search_maps.py --metadata_csv ../val2025_map_data.csv --search3d --rotation_resol 1 --chunk_size 2 --postfilter --num_translations_pf 1 --translation_extent_pf 0 --postfilter_chunk_size 1 --output_dir /home/gridsan/jroney/search_results2
