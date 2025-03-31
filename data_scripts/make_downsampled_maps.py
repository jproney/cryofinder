import pandas as pd
from cryodrgn import mrcfile as mrc
import torch, os
from search3d import downsample_vol

all_vols = []
all_ids = []
dat = pd.read_csv("/home/gridsan/jroney/siren_vols.pt")

for i,(m, apix) in enumerate(zip(dat['map_name'], dat['raw_pixel_size_angstrom'])):
    f = "../raw_maps_07072023/" + m
    if os.path.exists(f):
        vol, _ = mrc.parse_mrc(f)
        vol = downsample_vol(torch.tensor(vol), apix, target_res=5)
        all_vols.append(vol)
        all_ids.append(int(m.split("_")[-1][:-4]))
        if i % 10 == 0:
            print(i)

all_vols = torch.stack(all_vols, dim=0)
torch.save({"vols" : all_vols, "ids" : torch.tensor(all_ids)},  "../siren_vols.pt")