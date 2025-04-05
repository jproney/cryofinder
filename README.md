# Cryofinder: Similarity search for 2D and 3D CryoEM Data

Cryofinder searches EMDB for entries that are similar a query volume or class average. For more detail on the project, see [this presentation](https://docs.google.com/presentation/d/1hc0vHot9foLLG8RxMl-MbqJ3bkJ6xuzg/edit?usp=share_link&ouid=105615752929398207186&rtpof=true&sd=true).

## Installation

To install this project:
* `git clone https://github.com/jproney/cryofinder.git`
* `cd cryofinder`
* `pip install -e .` 

The most important dependency of this project is [CryoDRGN](https://github.com/ml-struct-bio/cryodrgn), as well as it's supporting packages (listed in requirements.txt)

## Repository Structure

- `cryofinder/` -- contains core resposity code
  - `search2d.py` -- functions for comparing 2d projections
  - `search3d.py` -- functions for comparing 3d volumes
- `data_scripts/` -- containts scripts for generating the reference and training data sets
- `notebooks/` -- notebooks for visualizing data and results
- `resnet/` -- code for the CNN embedding mdoels
  - `data.py` -- dataset for positive and negative contrastive pairs
  - `train.py` -- Pytorch Lightning script for training the embedding model
- `run_search.py` -- top-level script for executing 2d and 3d searches
- `slurm_scripts/` -- sbatch scripts to run searches (configured for MIT Supercloud)

## Datasets 

To run 2d searches, you first need to generate a dataset of reference projections. Here's an example of how to build one from scratch:
* `python data_scripts/download_emdb.py my_emdb_ids.txt --output_dir /path/to/downloaded/volumes/ --output_csv my_emdb_metadata.csv`
* `python data_scripts/batch_project.py my_emdb_metadata.csv /path/to/downloaded/volumes/ /path/to/projections/` (will be much faster on a GPU)
* `python data_scripts/make_projection_dataset.py --input_file my_emdb_ids.txt --projections_dir /path/to/projections --output_file my_proj_dataset.pt`

For examples of `my_emdb_ids.txt` and `my_emdb_metadata.csv` see `assets/`. The metadata csv used for building the SIREn set is in `assets/siren_set_metadata.csv`

For generating queries you can use the same workflow, execpt if your map is not in the EMDB you need to create the metadata csv manually. The last step (creating a .pt data file) is not neccesary for generating a query.

## Running a Search

After doing the above you can run a search! Do so by running
`run_search.py --metadata_csv query_metadata.csv --query_dir /path/to/query/projections --output_dir /path/to/search/results --search_data my_proj_dataset.pt` (Also requires GPU)

## Analyzing Results

To analyze the results of the search, take a look at the notebooks in `/notebooks` to get started.



