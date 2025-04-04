# Cryofinder: Similarity search for 2D and 3D CryoEM Data

Cryofinder searches EMDB for entries that are similar a query volume or class average. For more detail on the project, see [this presentation](https://docs.google.com/presentation/d/1hc0vHot9foLLG8RxMl-MbqJ3bkJ6xuzg/edit?usp=share_link&ouid=105615752929398207186&rtpof=true&sd=true).

## Installation

To install this project:
* `git clone https://github.com/jproney/cryofinder.git`
* `cd cryofinder`
* `pip install -e .` 

The most important dependency of this project is [CryoDRGN](https://github.com/ml-struct-bio/cryodrgn), as well as it's supporting packages (listed in requirements.txt)

## Repository Structure

- `/cryofinder` -- contains core resposity code
  - `search2d.py` -- functions for comparing 2d projections
  - `search3d.py` -- functions for comparing 3d volumes
- `/data_scripts` -- containts scripts for generating the reference and training data sets
- `/notebooks` -- notebooks for visualizing data and results
- `/resnet` -- code for the CNN embedding mdoels
  - `data.py` -- dataset for positive and negative contrastive pairs
  - `train.py` -- Pytorch Lightning script for training the embedding model
- `run_search.py` -- top-level script for executing 2d and 3d searches
- `/slurm_scripts` -- sbatch scripts to run searches (configured for MIT Supercloud)

## Datasets 

To run 2d searches, you first need to generate a dataset of reference projections. To do so run the following:
* 


## Getting Started



