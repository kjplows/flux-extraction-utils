import json
from pathlib import Path
from collections import defaultdict

groups = defaultdict(list)

directory = Path("/pnfs/sbnd/scratch/users/kplows/analyse_beammc_voxels/analyse_beammc_voxels_500files_newVolume_03/work-products")

for path in directory.glob("*.json"):
    with path.open() as f:
        data = json.load(f)
        
        key = (
            data["x"],
            data["y"],
            data["first_file_index"],
            data["last_file_index"],
        )
        
        groups[key].append(path)
        
for key, files in groups.items():
    x0, y0, f0, f1 = key
            
    for filename in files:
        print(f"{filename} {x0} {y0} {f0} {f1}")
