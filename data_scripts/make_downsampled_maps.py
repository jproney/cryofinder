import pandas as pd
from cryodrgn import mrcfile as mrc
import torch, os
import torch.nn.functional as F


def downsample_vol(map, res, target_res=5, target_size=128):
    """
    Downsample a DxDxD image to a target resolution and pad to target size.

    Args:
        map: Tensor of shape D x D x D, input volume
        res: Current resolution of the input volume in A/pix
        target_res: Target resolution in A/pix (default is 5 A/pix)
        target_size: Target size for padding (default is 128)

    Returns:
        Tensor of shape target_size x target_size x target_size, downsampled and padded volume
    """
    if res < target_res:
        scale_factor = res / target_res
        D = map.shape[0]
        new_D = int(D * scale_factor)
        grid = torch.meshgrid([torch.linspace(-1, 1, new_D) for _ in range(3)], indexing='ij')
        grid = torch.stack(grid, dim=-1).unsqueeze(0).to(map.device)
        map = F.grid_sample(map.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze()

    pad_width = sum([((target_size - s) // 2, (target_size - s + 1) // 2) for s in map.shape], ())
    map = F.pad(map, pad_width, mode='constant', value=0)

    return map


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