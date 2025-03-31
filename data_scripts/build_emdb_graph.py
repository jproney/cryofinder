import argparse, os

# Set up argument parser
parser = argparse.ArgumentParser(description='Build EMDB graph from annotation files.')
parser.add_argument('--input_dir', type=str, default="/home/gridsan/jroney/emdb_graph/",
                    help='Directory containing EMDB annotation files')
parser.add_argument('--output_file', type=str, default="/home/gridsan/jroney/emdb_graph_data_seqdata_union.pkl",
                    help='Output pickle file path')
parser.add_argument('--distance_mat_ids', type=str, default="/home/gridsan/jroney/ids_all.pt",
                    help='file with list of ids to include in the final distnace matrix')
parser.add_argument('--output_file_dmat', type=str, default="/home/gridsan/jroney/siren_vols_distance_mat.pt",
                    help='Output pickle file for the final distance matrix')
args = parser.parse_args()

# only use a subset that are relevant to our desired clusterings. EMDB just uses interpro and go
files = ['emdb_interpro.tsv',
         #'emdb_model.tsv',
         #'emdb_drugbank.tsv',
         'emdb_uniprot.tsv',
         'emdb_pfam.tsv',
         #'emdb_pubmed.tsv',
         'emdb_scop2.tsv',
         #'emdb_orcid.tsv',
         'emdb_alphafold.tsv',
         'emdb_cath.tsv',
        # 'emdb_scop.tsv',
         'emdb_pdbekb.tsv',
         #'emdb_author.tsv',
         'emdb_go.tsv',
         #'emdb_chembl.tsv',
         #'emdb_cpx.tsv',
         #'emdb_chebi.tsv',
         'emdb_scop2B.tsv',
         'emdb_rfam.tsv']
         #'emdb_empiar.tsv']


keys = [f[5:-4] for f in files]

emdb_dict = {}

for i,f in enumerate(files):
    for l in open(os.path.join(args.input_dir, f)).readlines()[1:]:
        fields = l.split('\t')
        emd_id = fields[0]
        if f in ['emdb_model.tsv', 'emdb_pubmed.tsv', 'emdb_empiar.tsv']:
            db_id = fields[1]
        elif f == 'emdb_cpx.tsv':
            db_id = fields[4]
        elif f in ['emdb_uniprot.tsv', 'emdb_chembl.tsv', 'emdb_drugbank.tsv', 'emdb_chebi.tsv']:
            db_id = fields[5]
        else:
            db_id = fields[2]

        if emd_id not in emdb_dict:
            emdb_dict[emd_id] = {k : [] for k in keys}

        if len(db_id) > 0:
            emdb_dict[emd_id][keys[i]].append(db_id)

import numpy as np

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0, union

def compute_similarity(dict1, dict2):
    """Compute average Jaccard similarity between two dictionaries."""
    keys = dict1.keys()
    similarities = [jaccard_similarity(set(dict1[k]), set(dict2[k])) for k in keys]
    return sum(similarities) / len(keys)  # Averaging over all keys

sim_matrix = np.zeros([len(emdb_dict), len(emdb_dict), len(keys)])
union_matrix = np.zeros([len(emdb_dict), len(emdb_dict), len(keys)])
print(f"matrix has dim {len(emdb_dict)}")

keylist = list(emdb_dict.keys())
for i in range(len(keylist)):
    if i % 1000 == 0:
        print(i)

    sim_matrix[i,i] = 1.0
    for j in range(i):
        e1 = emdb_dict[keylist[i]]
        e2 = emdb_dict[keylist[j]]
        for ki, k, in enumerate(keys):
            sim, union = jaccard_similarity(set(e1[k]), set(e2[k]))
            sim_matrix[i,j,ki] = sim
            sim_matrix[j,i,ki] = sim
            union_matrix[i,j,ki] = union
            union_matrix[j,i,ki] = union

import pickle
pickle.dump({"sim_matrix" : sim_matrix, "keys" : keylist, 'sources' : keys, 'union' : union_matrix}, open(args.output_file, 'wb'))


ids_all = [int(x[:-1].split("_")[-1]) for x in open(args.distance_mat_ids,'r').readlines()]


# make distance metric according to the exact recipe from the EMDB people
ip = keys.index('interpro')
go = keys.index('go')
emdb_graph = (sim_matrix[..., ip] * union_matrix[...,ip] + sim_matrix[..., go] * union_matrix[...,go]) / (np.maximum(union_matrix[...,ip] + union_matrix[...,go], 1))

def find_index(list_data, element):
    try:
        return list_data.index(element)
    except ValueError:
        return None

ids_all = []
idx_to_key = [find_index(keylist, f"EMD-{i.item()}") for i in ids_all]
idx_to_key_flitered = [x for x in idx_to_key if x]
idx_flitered = [i for i,x in enumerate(idx_to_key) if x]

dmat = np.zeros([ids_all.shape[0], ids_all.shape[0]])
dmat[np.ix_(idx_flitered, idx_flitered)] = emdb_graph[np.ix_(idx_to_key_flitered, idx_to_key_flitered)]

np.fill_diagonal(dmat, 1.0)

import torch

dmat = torch.tensor(dmat)

torch.save(dmat, args.output_file_dmat)
