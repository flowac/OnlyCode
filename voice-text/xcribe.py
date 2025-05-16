import sys
import whisper

flist = ["sample1.wav", "sample2.wav"]

if len(sys.argv) > 1:
    flist = []
    for fname in sys.argv[1:]:
        flist.append(fname)

model = whisper.load_model("tiny.en")

for idx, fname in enumerate(flist):
    print("{}. {}:".format(idx, fname))
    result = model.transcribe(fname)
    print(result["text"])

