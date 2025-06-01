import sys
import time
import pose
import json

flist = ["data/image.png", "data/image.jpg", "data/bad.png"]
if len(sys.argv) > 1:
    flist = sys.argv[1:]
results = []
po = pose.pose()

for fname in flist:
    start_time = time.time()
    fbs = po.run(fname)
    stop_time = time.time()
    print(f'\n{fname} took {(stop_time - start_time) * 1000} ms')

    if fbs is None:
        print("No features detected")
        continue

    result = {
        'filename': fname,
        'features': fbs  # This is a dictionary of feature name → value
    }

    results.append(result)
    for name, val in fbs.items():
        print(f'{name: <20} {val}')

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
