import pandas as pd
from cryodrgn import mrcfile as mrc
import torch, os
from map_search import downsample_vol

all_vols = []
all_ids = []
dat = pd.read_csv("../siren_maps.csv")

for i,(m, apix) in enumerate(zip(dat['map_name'], dat['raw_pixel_size_angstrom'])):
    f = "../raw_maps_07072023/" + m
    if os.path.exists(f):
        vol, _ = mrc.parse_mrc(f)
        vol = downsample_vol(torch.tensor(vol), apix, target_res=5)
        all_vols.append(vol)
        all_ids.append(int(m.split("_")[-1][:-4]))
        if i % 10 == 10:
            print(i)

all_vols = torch.cat(all_vols)
torch.save({"vols" : all_vols, "ids" : torch.tensor(all_ids)},  "../siren_vols.pt")