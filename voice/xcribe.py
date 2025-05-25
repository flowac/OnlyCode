import sys, time
import whisper
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

# input should be at least 30 second segments
# maybe 1 or 2 minute chunks

flist = ["terry.mp3", "micro-machine.wav"]
top_k = 5
top_thres = 0.05
mood_filter = {"neutral"}
pos_filter = {"admiration", "amusement", "approval", "caring", "curiosity", "desire", "excitement",
              "gratitude", "joy", "love", "optimism", "pride", "realization", "relief"}

if len(sys.argv) > 1:
    flist = sys.argv[1:]

voice2text = whisper.load_model("tiny.en")
text_model = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(text_model)
summarizer = BartForConditionalGeneration.from_pretrained(text_model)
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Score calculation strategy:
# filter out neutral
# balance of positive and negative emotions
# if only positive then it means no conflict / no learning
# if only negative then did not solve problem
def score(word_count, word_percent, pos_count, neg_count):
    density = "low"
    impact = "low"
    if word_count < 20:
        return None, None

    if word_percent > 15:
        density = "good"
        if word_percent > 30:
            density = "exceptional"

    pn_total = pos_count + neg_count
    if pos_count >= 1 and neg_count >= 1:
        impact = "medium"
        if pn_total >= 4:
            impact = "high"

    return density, impact

def transcribe(fname):
    result = voice2text.transcribe(fname)
    word_count_total = len(result["text"].split())
    print(f'\n{fname}: {word_count_total} words')

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
    word_count_summary = len(summary.split())
    word_percent = word_count_summary * 100 / word_count_total
    print(f'Summary: {word_count_summary} words, {word_percent} percent of total')
    print(summary)

    idata = [result["text"]]
    odata = classifier(idata) 
    cnt = 0
    pos_count = 0
    neg_count = 0
    for entry in odata[0]:
        if entry["label"] not in mood_filter:
            cnt = cnt + 1
            if cnt > top_k or entry["score"] < top_thres:
                break
            print(entry)
            if entry["label"] in pos_filter:
                pos_count = pos_count + 1
            else:
                neg_count = neg_count + 1

    density, impact = score(word_count_total, word_percent, pos_count, neg_count)
    print(f'Result: density = {density}, impact = {impact}')

for fname in flist:
    start_time = time.time()
    transcribe(fname)
    stop_time = time.time()
    print(f'{(stop_time - start_time) * 1000} ms')

