import json
from pathlib import Path
from collections import defaultdict

directory = Path("/pnfs/sbnd/scratch/users/kplows/analyse_beammc_voxels_500files_00/work-products")

groups = defaultdict(list)

for path in directory.glob("*.json"):
    with path.open() as f:
        data = json.load(f)
        
        # Canonical representation so dict key ordering doesn't matter
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        groups[key].append(path)
    
for files in groups.values():
    if len(files) > 1:
        files.sort(key=lambda p: p.stat().st_mtime)

        for f in files[:-1]:
            print(f)
        
        #print("Duplicate group:")
        #for f in files:
        #    print(f"  {f}  ({f.stat().st_mtime})")
                
        #print(f"  MOST RECENT: {files[-1]}")
        #print()
