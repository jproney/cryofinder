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
python ../run_search.py --metadata_csv /home/gridsan/jroney/test_cases/vos_data.csv --map_dir /home/gridsan/jroney/test_cases  --rotation_resol 1 --chunk_size 2400 --postfilter --num_translations_pf 7 --translation_extent_pf 7 --rotation_resol_pf 2 --pf_all_proj --postfilter_chunk_size 30 --output_dir /home/gridsan/jroney/search_results_refac
