import sys, time
import whisper
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

flist = ["terry.mp3", "micro-machine.wav"]
top_k = 5
top_thres = 0.05
mood_filter = ["neutral"]

if len(sys.argv) > 1:
    flist = sys.argv[1:]

voice2text = whisper.load_model("tiny.en")
text_model = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(text_model)
summarizer = BartForConditionalGeneration.from_pretrained(text_model)
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def transcribe(fname):
    print(f'\n{fname}:')
    result = voice2text.transcribe(fname)
    #print(result["text"])

    inputs = tokenizer(result["text"], return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = summarizer.generate(
        inputs.input_ids,
        num_beams=4,              # Beam search for higher quality
        max_length=130,           # Maximum summary length
        min_length=30,            # Minimum summary length
        early_stopping=True,      # Stop early if possible
        no_repeat_ngram_size=3    # Avoid repeating 3-grams
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f'Summary: {summary}')

    idata = [result["text"]]
    odata = classifier(idata) 
    cnt = 0
    for entry in odata[0]:
        if entry["label"] not in mood_filter:
            cnt = cnt + 1
            if cnt > top_k or entry["score"] < top_thres:
                break
            print(entry)

# proposed score calculation:
# balance of positive and negative emotions
# if only positive then it means no conflict / no learning
# if only negative then did not solve problem
# maybe filter out neutral

for fname in flist:
    start_time = time.time()
    transcribe(fname)
    stop_time = time.time()
    print(f'{(stop_time - start_time) * 1000} ms')

