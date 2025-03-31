# Now let's try doing it the right way with the EMDB graph thing
import os

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
        # 'emdb_scop.tsv', -- this one is empty
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
    for l in open("../emdb_graph/" + f).readlines()[1:]:
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

from itertools import combinations
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
        with open('emdb_graph_log.txt','a') as f:
            f.write(f'{i}\n')

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
pickle.dump({"sim_matrix" : sim_matrix, "keys" : keylist, 'sources' : keys, 'union' : union_matrix}, open("../emdb_graph_data_seqdata_union.pkl", 'wb'))
