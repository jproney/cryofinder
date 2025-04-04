import os
import argparse
import subprocess
import pandas as pd

parser = argparse.ArgumentParser(description='Batch process .map files with project3d.py')
parser.add_argument('input_file', default="/home/gridsan/jroney/siren_maps.csv", help='csv with the names of volumes and metadata')
parser.add_argument('input_dir', default="/home/gridsan/jroney/raw_maps_07072023/", help='Directory containing .map files')
parser.add_argument('output_dir', default="/home/gridsan/jroney/projections/", help='Directory for output files')
parser.add_argument('--start', type=int, default=0, help='Lowest index to process')
parser.add_argument('--end', type=int, default=10000, help='Highest index to process')
parser.add_argument('--healpy-grid', type=int, default=2, help='Resolution level for healpy grid')
args = parser.parse_args()

dat = pd.read_csv(args.input_file)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

maplist = dat['map_name'][args.start:args.end]
apixlist = dat['raw_pixel_size_angstrom'][args.start:args.end]
sizelist = dat['raw_box_size_pixel'][args.start:args.end]

print(f"Found {len(maplist)} .map files")

# Process each map file
for map_name, apix, dim in zip(maplist, apixlist, sizelist):


    input_path = os.path.join(args.input_dir, map_name) + ".map"

    # Setup output paths
    outstack = os.path.join(args.output_dir, f"{map_name}.mrcs")
    out_pose = os.path.join(args.output_dir, f"{map_name}_pose.pkl")
    
    # Skip if output files already exist
    if os.path.exists(outstack) and os.path.exists(out_pose):
        print(f"\nSkipping {map_name} - output files already exist")
        continue
    
    # Build command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        'python', 
        os.path.join(script_dir, 'project3d.py'),
        input_path,
        outstack,
        '--apix',
        str(apix),
        '--healpy-so2-grid', 
        str(args.healpy_grid),
        '--out-pose',
        out_pose
    ]
    
    print(f"\nProcessing {map_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {map_name}: {e}")
        continue
        
    print(f"Completed processing {map_name}")
