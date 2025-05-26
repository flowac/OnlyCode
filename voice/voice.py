import os.path
import whisper
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

class voice:
    top_k = 5
    top_thres = 0.05
    mood_filter = {"neutral"}
    pos_filter = {"admiration", "amusement", "approval", "caring", "curiosity", "desire", "excitement",
                  "gratitude", "joy", "love", "optimism", "pride", "realization", "relief"}

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
    def score(self, word_count, word_percent, pos_count, neg_count):
        density = "low"
        impact = "low"
        if word_count < 30:
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

        print(f'Result: density = {density}, impact = {impact}')
        return density, impact

    def run(self, fname):
        try:
            return self._run(fname)
        except:
            return None, None

    # do not call directly
    def _run(self, fname):
        if not os.path.isfile(fname):
            return None, None

        result = self.voice2text.transcribe(fname)
        word_count_total = len(result["text"].split())
        print(f'\n{fname}: {word_count_total} words')

        inputs = self.tokenizer(result["text"], return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = self.summarizer.generate(
            inputs.input_ids,
            num_beams=4,              # Beam search for higher quality
            max_length=130,           # Maximum summary length
            min_length=30,            # Minimum summary length
            early_stopping=True,      # Stop early if possible
            no_repeat_ngram_size=3    # Avoid repeating 3-grams
        )

        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        word_count_summary = len(summary.split())
        word_percent = word_count_summary * 100 / word_count_total
        print(f'Summary: {word_count_summary} words, {word_percent} percent of total')
        print(summary)

        idata = [result["text"]]
        odata = self.classifier(idata) 
        cnt = 0
        pos_count = 0
        neg_count = 0
        for entry in odata[0]:
            if entry["label"] not in self.mood_filter:
                cnt = cnt + 1
                if cnt > self.top_k or entry["score"] < self.top_thres:
                    break
                print(entry)
                if entry["label"] in self.pos_filter:
                    pos_count = pos_count + 1
                else:
                    neg_count = neg_count + 1

        return self.score(word_count_total, word_percent, pos_count, neg_count)

