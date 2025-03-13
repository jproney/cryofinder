import torch
import numpy as np
from cryodrgn.source import ImageSource
import os
import pickle
import torch
import argparse

dat = torch.load("../train_projections.pt")
ids = dat['ids']
images = dat['images']
phis = dat['phis']
thetas = dat['thetas']

dat_val = torch.load("../val_projections.pt")
ids_val = dat_val['ids']
images_val = dat_val['images']
phis_val = dat_val['phis']
thetas_val = dat_val['thetas']

images_all_raw = torch.cat([images, images_val]).view((-1,128,128))
ids_all = torch.cat([ids, ids_val]).unsqueeze(-1).expand([-1, 192]).reshape((-1,))


from cryodrgn.source import ImageSource
import os
import torch

emds = [x[:-1].lower().replace('-','_') for x in open("../val2025_maps.txt",'r').readlines()]
query_imglist = []
for i,e in enumerate(emds):
    if os.path.exists('/home/gridsan/jroney/val_2025_dataset/' + e + ".mrcs"):
        query_imglist.append(ImageSource.from_file('/home/gridsan/jroney/val_2025_dataset/' + e + ".mrcs").images())

query_imgs_raw = torch.stack(query_imglist)


from proj_search import optimize_theta_trans_chunked

from cryodrgn import shift_grid, so3_grid

# Set up argument parser
parser = argparse.ArgumentParser(description='Search maps with specified rotations and translations.')
parser.add_argument('--rotation_resol', type=int, default=2, help='Number of rotations to use')
parser.add_argument('--num_translations', type=int, default=1, help='Number of translations to use')
parser.add_argument('--translation_extent', type=int, default=0, help='Extent of the translations')
parser.add_argument('--chunk_size', type=int, default=1280, help='Chunk size for optimization')
parser.add_argument('--fast_rotate', action="store_true", help="use fast rotation thing from cryodrgn")
parser.add_argument('--realspace_corr', action="store_true", help="compute correlation in real space instead of hartley")

args = parser.parse_args()

# Use command-line arguments for the number of rotations, translations, translation extent, and chunk size
rotation_resol = args.rotation_resol
num_translations = args.num_translations
translation_extent = args.translation_extent
chunk_size = args.chunk_size

# Generate translation and rotation grids based on the parameters
if translation_extent == 0 or num_translations == 1:
    trans = None
else:
    trans = torch.tensor(shift_grid.base_shift_grid(0, translation_extent, num_translations, xshift=0, yshift=0))

angles = torch.tensor(so3_grid.grid_s1(rotation_resol), dtype=torch.float)

import time

for query_batch, e in zip(query_imgs_raw, emds):
    print(f"running queries for {e}")
    output_file_name = f'/home/gridsan/jroney/val_2025_dataset/{e}_search_res_rot{rotation_resol}_trans{num_translations}_extent{translation_extent}'
    if not args.fast_rotate:
        output_file_name = output_file_name + "_slowrotate"
    if args.realspace_corr:
        output_file_name = output_file_name + "_realspace"
    output_file_name = output_file_name + ".pt"

    if os.path.exists(output_file_name):
        continue

    query_batch = query_batch.cuda(non_blocking=True)  # Move to CUDA before computation

    start_time = time.time()
    with torch.no_grad():  # Prevent unnecessary gradient tracking
        best_corr, best_indices, corr = optimize_theta_trans_chunked(
            (images_all_raw - images_all_raw.mean(dim=(-1,-2), keepdim=True)).view([-1,128,128]), 
            (query_batch - query_batch.mean(dim=(-1,-2), keepdim=True)), 
            trans.cuda(), 
            angles, 
            chunk_size=chunk_size,
            fast_rotate=args.fast_rotate, 
            hartley_corr= not args.realspace_corr
        )
    end_time = time.time()
    search_time = end_time - start_time
    print(f"Search took {search_time:.2f} seconds")

    # Save the results including rotation and translation vectors and timing
    torch.save({
        "best_corr": best_corr.cpu(), 
        "best_indices": best_indices.cpu(), 
        "corr": corr.cpu(),
        "rotation_vectors": angles.cpu(),  # Save rotation vectors
        "translation_vectors": trans.cpu(),  # Save translation vectors
        "search_time": search_time  # Save the timing information
    }, output_file_name)

    # Cleanup
    del query_batch, best_corr, best_indices, corr
    torch.cuda.empty_cache()
