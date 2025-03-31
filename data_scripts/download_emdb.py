import os
import requests
import gzip
import mrcfile
import shutil

def get_emdb_metadata(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
            return response.json()
    return {}



def download_emdb_entry(emdb_id, output_dir="/home/gridsan/jroney/val_2025_dataset"):
    emdb_id = emdb_id.split('-')[-1]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download density map (if exists)
    map_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
    print(map_url)
    map_path = os.path.join(output_dir, f"emd_{emdb_id}.map.gz")
    extracted_map_path = os.path.join(output_dir, f"emd_{emdb_id}.map")

    map_response = requests.get(map_url)
    if map_response.status_code == 200:
        with open(map_path, "wb") as f:
            f.write(map_response.content)
        print(f"Downloaded density map: {map_path}")

        # Decompress .gz file
        with gzip.open(map_path, 'rb') as f_in, open(extracted_map_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted map: {extracted_map_path}")
        return extracted_map_path
    else:
        print(f"Density map not found for EMD-{emdb_id}")

    
def load_map(map_path):
    if not os.path.exists(map_path):
        print(f"File not found: {map_path}")
        return None
    
    with mrcfile.open(map_path, permissive=True) as mrc:
        data = mrc.data.copy()
        print(f"Loaded map with shape: {data.shape}")
        return data



csv_lines = ["emdb_map_file,raw_pixel_size_angstrom,raw_box_size_pixel\n"]
embds = [x[:-1] for x in open('val2025_maps.txt', 'r').readlines()]

for e in embds:
    path = download_emdb_entry(e)
    meta = get_emdb_metadata(e)
    pix_size = meta['map']['pixel_spacing']['x']['valueOf_']
    dim = meta['map']['dimensions']['col']

    csv_lines.append(f"{path},{pix_size},{dim}\n")

with open("val2025_map_data.csv", 'w') as f:
    f.writelines(csv_lines)