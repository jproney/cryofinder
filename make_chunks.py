import pickle
from cryodrgn.source import ImageSource
import os
import torch

bsize = 250
emds = list(set([x[:-5] for x in os.listdir("/nobackup/users/jamesron/projections/") if x.startswith("emd") and x.endswith("mrcs")]))


for i in range(0, len(emds), bsize):
    imglist = []
    poselist = []
    ids = []

    for e in emds[i:i+bsize]:
        imglist.append(ImageSource.from_file("/nobackup/users/jamesron/projections/" + e + ".mrcs").images())
        poselist.append(pickle.load(open("/nobackup/users/jamesron/projections/" + e + "_pose.pkl", 'rb')))
        ids.append(int(e.split("_")[-1]))

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(-1).expand([-1, imglist[0].shape[0]]).reshape(-1)
    imglist = torch.stack(imglist, dim=0).view((-1, *imglist[0].shape[1:]))
    phis = torch.tensor([x[0] for x in poselist]).view(-1)
    thetas = torch.tensor([x[1] for x in poselist]).view(-1)

    # Create dictionary with all data
    chunk_data = {
        'ids': ids,
        'images': imglist,
        'phis': phis, 
        'thetas': thetas
    }

    # Save to pickle file with range in filename
    outfile = f"/nobackup/users/jamesron/projection_chunks/chunk_{i}_{min(i + bsize, len(emds))}.pkl"
    print()
    with open(outfile, 'wb') as f:
        pickle.dump(chunk_data, f)