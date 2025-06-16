import sys, time
import count

flist = ["data/test1.png", "data/code.png", "data/code-gray.png", "data/code-scale13.png"]
if len(sys.argv) > 1:
    flist = sys.argv[1:]

wc = count.count()
total_start = time.time()

for fname in flist:
    start_time = time.time()
    val = wc.run(fname)
    stop_time = time.time()
    print(f'{fname: >25} {val: >5} words detected {(stop_time - start_time) * 1000} ms')

total_time = time.time() - total_start
print(f'\nTests took {total_time * 1000} ms overall')

