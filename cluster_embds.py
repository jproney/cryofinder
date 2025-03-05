import requests

# Function to fetch metadata for an EMDB entry
def get_emdb_metadata(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
            return response.json()
    return {}

all_ids = ["EMD-" + x[:-1].split("_")[-1] for x in open('train_ids.txt').readlines()] + ["EMD-" + x[:-1].split("_")[-1] for x in open('val_ids.txt').readlines()]

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


import google.generativeai as genai

# Set up Gemini API key
genai.configure(api_key="AIzaSyD_KvzoihsuuzmuclSYtqzOzlhxp6_toAw")

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash")
import pickle

dat = pickle.load(open("emdb_metadata.pkl", 'rb'))
ti = dat[0]['title']

def extract_proteins_genes(text):


    prompt = f"Extract the names of all biological entities from the input text. Extract specific molecule names (e.g. Myoglobin, 80S Ribosomal Subunit, miR16, cAMP, microtubules) and species names (e.g. SARS-CoV-2 Omicron BA.1, E. coli), but do not include highly generic terms like 'protein' and 'dna'. When in doubt, err on the side of including. The preceding entities are just examples, do not return these. Adopt a common format so that the list of names will always be the same even if minor difference exist in the input text. Return the results as a list. The input text from which you should extract entities is as follows:\n\n{text}\n\n"

    response = model.generate_content(prompt)

    return response.text


all_topics = pickle.load(open("emdb_topics.pkl", 'rb'))
import time

for i,d in enumerate(dat):
    if i  in [x[0] for x in all_topics]:
        continue 

    try:
        all_topics.append((i,extract_proteins_genes(d['name'] +' ' + d['title'])))
        if i % 10 == 0:
            print(i)

        time.sleep(0.5)
    except Exception as e:
        print("Hit limit, waiting 15s")
        time.sleep(15)

pickle.dump(all_topics, open("emdb_topics.pkl", 'wb'))