import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("sample1.wav")
print(result["text"])

result = model.transcribe("sample2.wav")
print(result["text"])

