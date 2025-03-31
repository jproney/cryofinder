import pickle
from cryodrgn.source import ImageSource
import torch


imglist = []
poselist = []
ids = []


import argparse

parser = argparse.ArgumentParser(description='Create projection dataset from image files.')
parser.add_argument('--input_file', type=str, default="/home/gridsan/jroney/all_ids.txt",
                    help='Text file containing list of image IDs')
parser.add_argument('--output_file', type=str, default="/home/gridsan/jroney/all_projections.pt",
                    help='Output file path for projection dataset')
args = parser.parse_args()

emds = [x[:-1] for x in open(args.input_file,'r').readlines()]

for i,e in enumerate(emds):

    imglist.append(ImageSource.from_file(e + ".mrcs").images())
    poselist.append(pickle.load(open(e + "_pose.pkl", 'rb')))
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