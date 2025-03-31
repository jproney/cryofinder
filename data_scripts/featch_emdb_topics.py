import requests
import pickle
import argparse

# Function to fetch metadata for an EMDB entry
def get_emdb_metadata(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
            return response.json()
    return {}


# Set up argument parser
parser = argparse.ArgumentParser(description='Fetch EMDB metadata and topics.')
parser.add_argument('--input_file', type=str, default="/home/gridsan/jroney/all_ids.txt",
                    help='Text file containing list of EMDB IDs')
parser.add_argument('--output_file', type=str, default="/home/gridsan/jroney/emdb_metadata.pkl",
                    help='Output pickle file for EMDB metadata')
args = parser.parse_args()


all_ids = ["EMD-" + x[:-1].split("_")[-1] for x in open(args.input_file).readlines()]

all_dats = []
for i, x in enumerate(all_ids):
    all_dats.append(get_emdb_metadata(x))
    if i % 10 == 0:
        print(i)

proc = []
for i,d in zip(all_ids, all_dats):
    proc.append({'emd_id' : i})
    if 'sample' in d:
        proc[-1]['name'] = d['sample']['name']['valueOf_']

        proc[-1]['organism'] = [m['natural_source'][0]['organism']['valueOf_'] for m in d['sample']['supramolecule_list']['supramolecule'] if 'natural_source' in m]

    if 'admin' in d and 'title' in d['admin']:
         proc[-1]['title'] = d['admin']['title']

pickle.dump(proc, open(args.output_file,'wb'))