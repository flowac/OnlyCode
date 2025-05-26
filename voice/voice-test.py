import sys, time
import voice

# input should be at least 30 second segments
# maybe 1 minute chunks

flist = ["data/terry.mp3", "data/micro-machine.wav", "data/bad.wav"]
if len(sys.argv) > 1:
    flist = sys.argv[1:]

vc = voice.voice()

for fname in flist:
    start_time = time.time()
    vc.run(fname)
    stop_time = time.time()
    print(f'{(stop_time - start_time) * 1000} ms')

