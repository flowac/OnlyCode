import sys, time
import count

flist = ["data/test1.png", "data/test2.png", "data/bad.png"]
if len(sys.argv) > 1:
    flist = sys.argv[1:]

wc = count.count()

for fname in flist:
    start_time = time.time()
    val = wc.run(fname)
    stop_time = time.time()
    print(f'{(stop_time - start_time) * 1000} ms')
    print(f'{val} words detected')

