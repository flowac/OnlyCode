import sys, time
import voice

# input should be at least 30 second segments
# maybe 1 minute chunks

flist = ["terry.mp3", "micro-machine.wav", "voice-test.py"]
if len(sys.argv) > 1:
    flist = sys.argv[1:]

vc = voice.voice()

for fname in flist:
    start_time = time.time()
    vc.run(fname)
    stop_time = time.time()
    print(f'{(stop_time - start_time) * 1000} ms')

