import pandas as pd
from cryodrgn import mrcfile as mrc
import torch, os
import torch.nn.functional as F
from cryofinder.search3d import downsample_vol


import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Downsample maps to a common resolution and size.')
parser.add_argument('--map_csv', type=str, default="/home/gridsan/jroney/siren_maps.csv", 
                    help='CSV file containing map metadata')
parser.add_argument('--map_dir', type=str, default="/home/gridsan/jroney/raw_maps_07072023",
                    help='Directory containing the input maps')
parser.add_argument('--output_file', type=str, default="/home/gridsan/jroney/siren_vols.pt",
                    help='Output file path for downsampled volumes')
args = parser.parse_args()

all_vols = []
all_ids = []
dat = pd.read_csv(args.map_csv)

for i,(m, apix) in enumerate(zip(dat['map_name'], dat['raw_pixel_size_angstrom'])):
    f = os.path.join(args.map_dir, m)
    if os.path.exists(f):
        vol, _ = mrc.parse_mrc(f)
        vol = downsample_vol(torch.tensor(vol), apix, target_res=5)
        all_vols.append(vol)
        all_ids.append(int(m.split("_")[-1][:-4]))
        if i % 10 == 0:
            print(i)

all_vols = torch.stack(all_vols, dim=0)
torch.save({"vols" : all_vols, "ids" : torch.tensor(all_ids)}, args.output_file)