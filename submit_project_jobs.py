import os

nmaps = len(open("siren_naps.csv"))
bsize = 100

for i in range(0, nmaps, bsize):

    print(f"sbatch project.sbatch {i} {i+bsize}")
    os.system(f"sbatch project.sbatch {i} {i+bsize}")