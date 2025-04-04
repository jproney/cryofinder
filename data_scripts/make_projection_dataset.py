import pickle
from cryodrgn.source import ImageSource
import torch
import os


imglist = []
poselist = []
ids = []


import argparse

parser = argparse.ArgumentParser(description='Create projection dataset from image files.')
parser.add_argument('--input_file', type=str, default="/home/gridsan/jroney/all_ids.txt",
                    help='Text file containing list of image IDs')
parser.add_argument('--projections_dir', type=str, default="",
                    help='Base directory containing projection files')
parser.add_argument('--output_file', type=str, default="/home/gridsan/jroney/all_projections.pt",
                    help='Output file path for projection dataset')
args = parser.parse_args()

emds = [x[:-1] for x in open(args.input_file,'r').readlines()]

for i,e in enumerate(emds):
    # Concatenate projections_dir with file paths
    e = e.replace("EMD-","emd_")
    file_path = os.path.join(args.projections_dir, e)
    
    imglist.append(ImageSource.from_file(file_path + ".mrcs").images())
    poselist.append(pickle.load(open(file_path + "_pose.pkl", 'rb')))
    ids.append(int(e.split("_")[-1]))
    print(i)

ids = torch.tensor(ids, dtype=torch.long)
imglist = torch.stack(imglist, dim=0)
phis = torch.tensor([x[0] for x in poselist])
thetas = torch.tensor([x[1] for x in poselist])

    # Create dictionary with all data
data = {
    'ids': ids,
    'images': imglist,
    'phis': phis, 
    'thetas': thetas
}

torch.save(data, args.output_file)