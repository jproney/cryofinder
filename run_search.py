import torch
import numpy as np
import os
import argparse
import pandas as pd
from cryodrgn import mrcfile as mrc
from cryodrgn.source import ImageSource
from cryofinder.search2d import optimize_theta_trans_chunked
from cryofinder.search3d import optimize_rot_trans_chunked, grid_3d, downsample_vol
from cryodrgn import shift_grid, so3_grid, lie_tools

# Set up argument parser
parser = argparse.ArgumentParser(description='Search maps with specified rotations and translations.')
parser.add_argument('--metadata_csv', type=str, required=True, help='Path to the CSV file containing metadata for the queries to run. \
                                                                Must have entries map_name, raw_box_size_pixel, raw_pixel_size_angstrom')
parser.add_argument('--query_dir', type=str, default='/home/gridsan/jroney/val_2025_dataset', help='Directory containing the query map or projection files')
parser.add_argument('--output_dir', type=str, default='/home/gridsan/jroney/search_results', help='Directory to save the search results')
parser.add_argument('--search3d', action="store_true", help="do search over 3d maps")
parser.add_argument('--search_data', type=str, default="", help='Directory containing the projections or volumes to search against')
parser.add_argument('--rotation_resol', type=int, default=1, help='Number of rotations to use (specified in terms of healpix resolution). \
                    These are S1 (i.e., in-plane) rotations for 2d search and SO(3) rotations (i.e., full rotation matrices) for 3d search')
parser.add_argument('--num_translations', type=int, default=1, help='Number of translations to use in each dimension \
                     (so total numbers is num_translations**2 or num_translations**3 for 2d and 3d search)')
parser.add_argument('--translation_extent', type=int, default=0, help='Extent of the translations in each dimension (in pixels)')
parser.add_argument('--chunk_size', type=int, default=2400, help='Chunk size for memory-efficient search')
parser.add_argument('--fast_rotate', action="store_true", help="use fast interpolation-based rotation thing from cryodrgn (only for 2d search)")
parser.add_argument('--realspace_corr', action="store_true", help="compute correlation in real space instead of hartley")
parser.add_argument('--mask_queries', action="store_true", help="mask the queries to only the actual particle region (currently only for 2d search)")
parser.add_argument('--clobber', action="store_true", help="Overwrite existing outputs")

# Add arguments for postfilter
parser.add_argument('--postfilter', action="store_true", help="Enable postfiltering of best densities")
parser.add_argument('--postfilter_num', type=int, default=64, help="Number of top densities to postfilter")
parser.add_argument('--translation_extent_pf', type=int, default=7, help="Extent of translations for postfiltering")
parser.add_argument('--num_translations_pf', type=int, default=7, help="Number of translations for postfiltering")
parser.add_argument('--rotation_resol_pf', type=int, default=2, help="Number of rotations for postfiltering")
parser.add_argument('--postfilter_chunk_size', type=int, default=30, help="Chunk size for postfiltering")
parser.add_argument('--pf_all_proj', action="store_true", help="Postfilter the top hits from each query projection, \
                                 not just the consensus top hits for queries that come from the same map (only applicable to 2d search)")


args = parser.parse_args()

if len(args.search_data) == 0:
    search_data = "/home/gridsan/jroney/siren_vols.pt" if args.search3d else "/home/gridsan/jroney/all_projections.pt"

if not args.search3d:
    dat = torch.load(search_data)
    ids = dat['ids']
    images = dat['images']
    phis = dat['phis']
    thetas = dat['thetas']

    ids_all = ids.unsqueeze(-1).expand([-1, 192]).reshape((-1,))

else:
    dat = torch.load(search_data)
    vols = dat['vols']
    ids = dat['ids']

# Use command-line arguments for the number of rotations, translations, translation extent, and chunk size
rotation_resol = args.rotation_resol
num_translations = args.num_translations
translation_extent = args.translation_extent
chunk_size = args.chunk_size

# Load metadata from CSV
dat = pd.read_csv(args.metadata_csv)
scaled_pix = torch.clamp(torch.ceil(torch.tensor(dat["raw_box_size_pixel"] * dat['raw_pixel_size_angstrom'] / 5)), None, 128)

# Extract IDs and paths from the CSV
query_imglist = []
for i, (e, apix) in enumerate(zip(dat['map_name'], dat['raw_pixel_size_angstrom'])):
    map_path = os.path.join(args.query_dir, f'{e}.map')
    mrcs_path = os.path.join(args.query_dir, f'{e}.mrcs')
    mrc_path = os.path.join(args.query_dir, f'{e}.mrc')

    if os.path.exists(map_path) and args.search3d:
        vol, _  = mrc.parse_mrc(map_path)
        vol = downsample_vol(torch.tensor(vol), apix, target_res=5).cuda()
        query_imglist.append(vol.unsqueeze(0))
    elif os.path.exists(mrcs_path):
        query_imglist.append(ImageSource.from_file(mrcs_path).images())
    elif os.path.exists(mrc_path):
        query_imglist.append(ImageSource.from_file(mrc_path).images())


query_imgs_raw = torch.stack(query_imglist)

y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
dist_squared = (x - 128//2) ** 2 + (y - 128//2) ** 2
radius_squared = (scaled_pix / 2).view(scaled_pix.shape[0], 1, 1) ** 2
circle_masks = (dist_squared.unsqueeze(0) < radius_squared)


# Generate translation and rotation grids based on the parameters
if translation_extent == 0 or num_translations == 1:
    trans = None
elif args.search3d:
    trans = torch.tensor(grid_3d(0,translation_extent, num_translations)).cuda()
else:
    trans = torch.tensor(shift_grid.base_shift_grid(0, translation_extent, num_translations, xshift=0, yshift=0)).cuda()

if args.search3d:
    quats1 = so3_grid.s2_grid_SO3(0).astype(np.float32)
    rots1 = lie_tools.quaternions_to_SO3(torch.from_numpy(quats1)).cuda()

    theta, phi = so3_grid.grid_s2(rotation_resol)
    psi = so3_grid.grid_s1(rotation_resol)
    quats2 = so3_grid.hopf_to_quat(
        np.repeat(theta, len(psi)),
        np.repeat(phi, len(psi)),
        np.tile(psi, len(theta)),
    ) 

    rots2 = lie_tools.quaternions_to_SO3(torch.from_numpy(quats2)).cuda()
else:
    angles = torch.tensor(so3_grid.grid_s1(rotation_resol), dtype=torch.float)

import time

for query_batch, e, m in zip(query_imgs_raw, dat['map_name'], circle_masks):
    print(f"running queries for {e}")
    output_file_name = os.path.join(
        args.output_dir,
        f'{e}_search_res'
        f'_rot{rotation_resol}_trans{num_translations}_extent{translation_extent}'
        f'{"_realspace" if args.realspace_corr else "_hartley"}' + ((
        f'{"_fastrotate" if args.fast_rotate else "_slowrotate"}'
        f'{"_maskqueries" if args.mask_queries else ""}')  if not args.search3d else '_3d') +
        ((f'_postfilternum{args.postfilter_num}'
        f'_transpf{args.translation_extent_pf}_numtranspf{args.num_translations_pf}'
        f'_rotpf{args.rotation_resol_pf}.pt') if args.postfilter else '.pt')
    )

    if os.path.exists(output_file_name) and not args.clobber:
        continue

    print(f"Writing results to {output_file_name}")


    query_batch = query_batch.cuda(non_blocking=True)

    start_time = time.time()
    with torch.no_grad():
        if args.search3d:
            best_corr, best_indices, corr = optimize_rot_trans_chunked(vols - vols.mean(dim=(-1,-2,-3), keepdim=True), 
                                              query_batch - query_batch.mean(dim=(-1,-2,-3), keepdim=True),
                                              rots1,
                                              rots2,
                                              trans,
                                              hartley_corr=not args.realspace_corr,
                                              chunk_size=chunk_size)
        else:
            best_corr, best_indices, corr = optimize_theta_trans_chunked(
                (images - images.mean(dim=(-1,-2), keepdim=True)).view([-1,128,128]), 
                (query_batch - query_batch.mean(dim=(-1,-2), keepdim=True)), 
                trans, 
                angles, 
                chunk_size=chunk_size,
                fast_rotate=args.fast_rotate, 
                hartley_corr= not args.realspace_corr,
                query_mask=m.unsqueeze(0).cuda() if args.mask_queries else None
            )

        if args.postfilter:
            corr = corr.cpu()
            if args.search3d:
                unique_indices = corr.amax(dim=(-1,-2)).topk(args.postfilter_num, dim=-1)[1][0] # elimintate uneended extra dimension 

                trans_pf = torch.tensor(grid_3d(0, args.translation_extent_pf, args.num_translations_pf)).cuda()
                theta, phi = so3_grid.grid_s2(args.rotation_resol_pf)
                psi = so3_grid.grid_s1(args.rotation_resol_pf)
                quats2 = so3_grid.hopf_to_quat(
                    np.repeat(theta, len(psi)),
                    np.repeat(phi, len(psi)),
                    np.tile(psi, len(theta)),
                ) 

                rots2_pf = lie_tools.quaternions_to_SO3(torch.from_numpy(quats2)).cuda()

                print(f"Postfiltering {unique_indices.shape[0]} densities...")

                vols_pf = vols[unique_indices]

                best_corr_pf, best_indices_pf, corr_pf = optimize_rot_trans_chunked(vols_pf - vols_pf.mean(dim=(-1,-2,-3), keepdim=True), 
                                                query_batch - query_batch.mean(dim=(-1,-2,-3), keepdim=True),
                                                rots1,
                                                rots2_pf,
                                                trans_pf,
                                                hartley_corr=not args.realspace_corr,
                                                chunk_size=args.postfilter_chunk_size)


            else:
                corr = corr.amax(dim=(-1,-2)) # don't care about the exact rotations / translations

                # For each combination of query projection and reference target, take the correlation of the 
                # best matching reference projection. Then average those values over the query projections to 
                # get an aggregated query-target fit score
                mean_best_per_query = corr.view([query_batch.shape[0],-1,192]).max(dim=-1)[0].mean(dim=0)
                mbq_indices = mean_best_per_query.topk(args.postfilter_num, dim=-1)[1]

                if args.pf_all_proj:
                    best_per_query_proj = corr.view([query_batch.shape[0],-1,192]).max(dim=-1)[0]
                    bpqp_indices = best_per_query_proj.topk(args.postfilter_num, dim=-1)[1]
                    unique_indices = torch.unique(torch.cat([mbq_indices.unsqueeze(0), bpqp_indices]))
                else:
                    bpqp_indices = None
                    unique_indices = torch.unique(mbq_indices)

                print(f"Postfiltering {unique_indices.shape[0]} densities...")

                pf_images_raw = images[unique_indices]

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

                corr_pf = corr_pf.amax(dim=(-1,-2))

    end_time = time.time()
    search_time = end_time - start_time
    print(f"Search took {search_time:.2f} seconds")

    res_dict = {"corr" : corr.cpu(),
                "translation_vectors" : None if trans is None else trans.cpu(),
                "search_time" : search_time,
                "rotation_vectors" : rots2.cpu() if args.search3d else angles,
                "corr_pf": corr_pf.cpu() if args.postfilter else None,
                "rotation_vectors_pf": (rots2_pf.cpu() if args.search3d else angles_pf.cpu()) if args.postfilter else None,
                "translation_vectors_pf": (None if trans_pf is None else trans_pf.cpu()) if args.postfilter else None,
                "best_corr": best_corr.cpu(), 
                "best_indices": best_indices.cpu(), 
                "best_corr_pf": best_corr_pf.cpu() if args.postfilter else None,
                "best_indices_pf": best_indices_pf.cpu() if args.postfilter else None,
                "unique_indices": unique_indices.cpu() if args.postfilter else None}

    if args.search3d:
        res_dict.update({
            "slice_rotations": rots1.cpu(), 
         })
    else:
        res_dict.update({
            "mbq_indices": mbq_indices.cpu() if args.postfilter else None,
            "bpqp_indices" : bpqp_indices.cpu() if args.pf_all_proj else None,
        })

    torch.save(res_dict, output_file_name)

    del query_batch, best_corr, best_indices, corr
    torch.cuda.empty_cache()
