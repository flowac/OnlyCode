import sys, time
import easyocr

class count:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def run(self, fname):
        try:
            return self._run(fname)
        except:
            return None

    def _run(self, fname):
        result = self.reader.readtext(fname)
        
        word_count = 0
        for _, text, _ in result:
            word_count += len(text.split())
        
        return word_count

