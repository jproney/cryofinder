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
        print(i)

        time.sleep(0.5)
    except Exception as e:
        print("Hit limit, waiting 15s")
        time.sleep(15)

pickle.dump(all_topics, open("emdb_topics.pkl", 'wb'))


import pickle
from fuzzywuzzy import fuzz
import numpy as np
import re
import math

dat = pickle.load(open("emdb_metadata.pkl", 'rb'))
all_topics = pickle.load(open("emdb_topics.pkl", 'rb'))


topics_proc = [None] * len(dat) 
dist_mat = np.zeros([len(dat), len(dat)])
all_topics = sorted(all_topics)


for i,t in all_topics:
    topics_proc[i]  = [x.lower() for x in re.split(r'[^a-zA-Z0-9().]{2,}|\n|[^\S ]', t.replace(',','').replace('json','')) if len(x) > 0 and x.count(' ') < 7 and x.lower() not in ['dna', 'rna', 'protein', 'complex']]

    # Remove duplicates and topics that are substrings of other topics
    topics_to_remove = set()
    unique_topics = list(dict.fromkeys(topics_proc[i]))  # Remove exact duplicates first
    
    for topic1 in unique_topics:
        for topic2 in unique_topics:
            if topic1 != topic2 and topic1 in topic2:
                topics_to_remove.add(topic1)
    
    topics_proc[i] = [t for t in unique_topics if t not in topics_to_remove]

    if len(topics_proc[i]) == 0:
        continue

    for j in range(i):
        if len(topics_proc[j]) == 0:
            continue

        dist_scores = np.zeros((len(topics_proc[i]), len(topics_proc[j])))
        # Calculate fuzzy match scores between all topic pairs
        for ti, topic1 in enumerate(topics_proc[i]):
            for tj, topic2 in enumerate(topics_proc[j]):
                dist_scores[ti,tj] = fuzz.ratio(topic1, topic2) / 100.0 * math.sqrt(len(topic1) + len(topic2))
                

        dist = (dist_scores.max(axis=1).mean() + dist_scores.max(axis=0).mean()) / 2
        dist_mat[i, j] = dist
        dist_mat[j, i] = dist  # Since the distance is symmetric

    print(i)

dist_mat[dist_mat < 0.5] = 0
dist_mat.mean()