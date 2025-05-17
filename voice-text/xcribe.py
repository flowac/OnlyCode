import sys
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration

flist = ["terry.mp3", "micro-machine.wav"]

if len(sys.argv) > 1:
    flist = []
    for fname in sys.argv[1:]:
        flist.append(fname)

voice2text = whisper.load_model("tiny.en")
text_model = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(text_model)
summarizer = BartForConditionalGeneration.from_pretrained(text_model)

for idx, fname in enumerate(flist):
    print("\n{}. {}:".format(idx, fname))
    result = voice2text.transcribe(fname)
    print(result["text"])

    print("\nSummary:")
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
    print(summary)

