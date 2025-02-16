import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Batch process .map files with project3d.py')
parser.add_argument('input_dir', help='Directory containing .map files')
parser.add_argument('output_dir', help='Directory for output files')
parser.add_argument('--healpy-grid', type=int, default=2, help='Resolution level for healpy grid')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Get list of .map files
map_files = [f for f in os.listdir(args.input_dir) if f.endswith('.map')]

if not map_files:
    print(f"No .map files found in {args.input_dir}")
    exit()
    
print(f"Found {len(map_files)} .map files")

# Process each map file
for map_file in map_files:
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
        '--healpy-grid', 
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
