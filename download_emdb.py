import os
import requests
import xml.etree.ElementTree as ET
import gzip
import mrcfile
import shutil

def download_emdb_entry(emdb_id, output_dir="emdb_data"):
    emdb_id = emdb_id.lower().replace("emdb-", "")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download XML metadata
    xml_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/header/emd-{emdb_id}.xml"
    xml_response = requests.get(xml_url)
    if xml_response.status_code != 200:
        print(f"Failed to download XML metadata for EMD-{emdb_id}")
        return
    
    # Parse XML
    root = ET.fromstring(xml_response.content)
    
    # Extract resolution
    resolution = None
    for res in root.findall(".//resolution"):  # Adjust path if needed
        resolution = res.text
        break
    
    # Extract PDB entries
    pdb_entries = []
    for pdb in root.findall(".//pdb_reference/pdb_id"):
        pdb_entries.append(pdb.text)
    
    # Download density map (if exists)
    map_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
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

    else:
        print(f"Density map not found for EMD-{emdb_id}")
    
    # Print summary
    print(f"EMD-{emdb_id}: Resolution = {resolution} Å, PDB entries = {pdb_entries}")
    
    return {
        "EMDB_ID": emdb_id,
        "Resolution": resolution,
        "PDB_Entries": pdb_entries,
        "Density_Map": extracted_map_path if os.path.exists(extracted_map_path) else None
    }


def load_map(map_path):
    if not os.path.exists(map_path):
        print(f"File not found: {map_path}")
        return None
    
    with mrcfile.open(map_path, permissive=True) as mrc:
        data = mrc.data.copy()
        print(f"Loaded map with shape: {data.shape}")
        return data


# Example usage
emdb_ids = ["1019", "3003"]
entries = []
for emdb_id in emdb_ids:
    entries.append(download_emdb_entry(emdb_id))
