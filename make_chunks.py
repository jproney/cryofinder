import pickle
from cryodrgn.source import ImageSource
import torch


imglist = []
poselist = []
ids = []


emds = [x[:-1] for x in open("val_ids.txt",'r').readlines()]

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

torch.save(data, "val_projections.pt")