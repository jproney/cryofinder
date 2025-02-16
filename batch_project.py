import os
import argparse
import subprocess
import pandas as pd

parser = argparse.ArgumentParser(description='Batch process .map files with project3d.py')
parser.add_argument('input_file', help='csv with the names of volumes and metadata')
parser.add_argument('input_dir', help='Directory containing .map files')
parser.add_argument('output_dir', help='Directory for output files')
parser.add_argument('--healpy-grid', type=int, default=2, help='Resolution level for healpy grid')
args = parser.parse_args()

dat = pd.read_csv(args.input_file)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
    
print(f"Found {len(dat['emdb_map_file'])} .map files")

# Process each map file
for map_file, apix in zip(dat['emdb_map_file'], dat['raw_pixel_size_angstrom']):
    input_path = os.path.join(args.input_dir, map_file)
    basename = os.path.splitext(map_file)[0]
    
    # Setup output paths
    outstack = os.path.join(args.output_dir, f"{basename}.mrcs")
    out_pose = os.path.join(args.output_dir, f"{basename}_pose.pkl")
    
    # Skip if output files already exist
    if os.path.exists(outstack) and os.path.exists(out_pose):
        print(f"\nSkipping {map_file} - output files already exist")
        continue
    
    # Build command
    cmd = [
        'python', 
        'project3d.py',
        input_path,
        outstack,
        '--apix',
        str(apix),
        '--healpy-so2-grid', 
        str(args.healpy_grid),
        '--out-pose',
        out_pose
    ]
    
    print(f"\nProcessing {map_file}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {map_file}: {e}")
        continue
        
    print(f"Completed processing {map_file}")
