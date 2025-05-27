import sys, time
import cv2
import pytesseract

class count:
    def run(self, fname):
        try:
            return self._run(fname)
        except:
            return None

    def _run(self, fname):
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--oem 3 --psm 11")

        return len(text.split())

