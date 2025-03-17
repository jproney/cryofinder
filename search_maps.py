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


import pandas as pd

dat = pd.read_csv("../val2025_map_data.csv")
scaled_pix = torch.clamp(torch.ceil(torch.tensor(dat["raw_box_size_pixel"] * dat['raw_pixel_size_angstrom'] / 5)), None, 128)

y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
dist_squared = (x - 128//2) ** 2 + (y - 128//2) ** 2
radius_squared = (scaled_pix / 2).view(scaled_pix.shape[0], 1, 1) ** 2
circle_masks = (dist_squared.unsqueeze(0) < radius_squared)



from proj_search import optimize_theta_trans_chunked

from cryodrgn import shift_grid, so3_grid

# Set up argument parser
parser = argparse.ArgumentParser(description='Search maps with specified rotations and translations.')
parser.add_argument('--rotation_resol', type=int, default=1, help='Number of rotations to use')
parser.add_argument('--num_translations', type=int, default=1, help='Number of translations to use')
parser.add_argument('--translation_extent', type=int, default=0, help='Extent of the translations')
parser.add_argument('--chunk_size', type=int, default=1280, help='Chunk size for optimization')
parser.add_argument('--fast_rotate', action="store_true", help="use fast rotation thing from cryodrgn")
parser.add_argument('--realspace_corr', action="store_true", help="compute correlation in real space instead of hartley")
parser.add_argument('--mask_queries', action="store_true", help="mask the queries to only the actual particle region")

# Add missing arguments for postfilter
parser.add_argument('--postfilter', action="store_true", help="Enable postfiltering of best densities")
parser.add_argument('--postfilter_num', type=int, default=64, help="Number of top densities to postfilter")
parser.add_argument('--translation_extent_pf', type=int, default=7, help="Extent of translations for postfiltering")
parser.add_argument('--num_translations_pf', type=int, default=7, help="Number of translations for postfiltering")
parser.add_argument('--rotation_resol_pf', type=int, default=2, help="Number of rotations for postfiltering")
parser.add_argument('--postfilter_chunk_size', type=int, default=1280, help="Chunk size for postfiltering")

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
    trans = torch.tensor(shift_grid.base_shift_grid(0, translation_extent, num_translations, xshift=0, yshift=0)).cuda()

angles = torch.tensor(so3_grid.grid_s1(rotation_resol), dtype=torch.float)

import time

for query_batch, e, m in zip(query_imgs_raw, emds, circle_masks):
    print(f"running queries for {e}")
    output_file_name = (
        f'/home/gridsan/jroney/search_results/{e}_search_res'
        f'_rot{rotation_resol}_trans{num_translations}_extent{translation_extent}'
        f'{"_fastrotate" if args.fast_rotate else "_slowrotate"}'
        f'{"_realspace" if args.realspace_corr else "_hartley"}'
        f'{"_maskqueries" if args.mask_queries else ""}'
        f'_postfilternum{args.postfilter_num}'
        f'_transpf{args.translation_extent_pf}_numtranspf{args.num_translations_pf}'
        f'_rotpf{args.rotation_resol_pf}.pt'
    )

    if os.path.exists(output_file_name):
        continue

    query_batch = query_batch.cuda(non_blocking=True)  # Move to CUDA before computation

    start_time = time.time()
    with torch.no_grad():  # Prevent unnecessary gradient tracking
        best_corr, best_indices, corr = optimize_theta_trans_chunked(
            (images_all_raw - images_all_raw.mean(dim=(-1,-2), keepdim=True)).view([-1,128,128]), 
            (query_batch - query_batch.mean(dim=(-1,-2), keepdim=True)), 
            trans, 
            angles, 
            chunk_size=chunk_size,
            fast_rotate=args.fast_rotate, 
            hartley_corr= not args.realspace_corr,
            query_mask=m.unsqueeze(0).cuda() if args.mask_queries else None
        )

        if args.postfilter: # postfilter the best densities by runnign them again
            corr = corr.cpu()
            mean_best_per_query = corr.view([query_batch.shape[0],-1,192]).max(dim=-1)[0].mean(dim=0)
            mbq_indices = mean_best_per_query.topk(args.postfilter_num, dim=-1)[1]

            pf_images_raw = images_all_raw[mbq_indices] # filter best densities

            trans_pf = torch.tensor(shift_grid.base_shift_grid(0, args.translation_extent_pf, args.num_translations_pf, xshift=0, yshift=0)).cuda()
            angles_pf = torch.tensor(so3_grid.grid_s1(args.rotation_resol_pf), dtype=torch.float)

            best_corr_pf, best_indices_pf, corr_pf = optimize_theta_trans_chunked(
                (pf_images_raw - pf_images_raw.mean(dim=(-1,-2), keepdim=True)).view([-1,128,128]), 
                (query_batch - query_batch.mean(dim=(-1,-2), keepdim=True)), 
                trans_pf, 
                angles_pf, 
                chunk_size=args.postfilter_chunk_size,
                fast_rotate=args.fast_rotate, 
                hartley_corr= not args.realspace_corr,
                query_mask=m.unsqueeze(0).cuda() if args.mask_queries else None
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
        "translation_vectors": None if trans is None else trans.cpu(),  # Save translation vectors
        "search_time": search_time,  # Save the timing information
        "best_corr_pf": best_corr_pf.cpu() if args.postfilter else None,  # Save postfilter best correlations
        "best_indices_pf": best_indices_pf.cpu() if args.postfilter else None,  # Save postfilter best indices
        "corr_pf": corr_pf.cpu() if args.postfilter else None,  # Save postfilter correlations
        "mbq_indices": mbq_indices.cpu() if args.postfilter else None,  # Save mbq indices
        "rotation_vectors_pf": angles_pf.cpu() if args.postfilter else None,  # Save postfilter rotation vectors
        "translation_vectors_pf": None if trans_pf is None else trans_pf.cpu() if args.postfilter else None  # Save postfilter translation vectors
    }, output_file_name)

    # Cleanup
    del query_batch, best_corr, best_indices, corr
    torch.cuda.empty_cache()
