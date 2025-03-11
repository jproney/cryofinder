import torch
import numpy as np
from cryodrgn.source import ImageSource
import os
import pickle
import torch

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
trans = torch.tensor(shift_grid.base_shift_grid(0, 7, 7, xshift=0, yshift=0))
angles = torch.tensor(so3_grid.grid_s1(2), dtype=torch.float)

all_res = []

for query_batch, e in zip(query_imgs_raw, emds):
    print(f"running queries for {emds}")
    best_corr, best_indices, corr = optimize_theta_trans_chunked((images_all_raw - images_all_raw.mean(dim=(-1,-2), keepdim=True)).view([-1,128,128]), (query_batch - query_batch.mean(dim=(-1,-2), keepdim=True)).cuda(), trans.cuda(), angles, chunk_size=30, fast_translate=False, fast_rotate=True, refine_fast_translate=False)
    torch.save({"best_corr" : best_corr, "best_indices" : best_indices, "corr" : corr}, '/home/gridsan/jroney/val_2025_dataset/' + e + "_search_res.pt")