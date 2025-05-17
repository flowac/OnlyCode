import sys, time
import cv2
import pytesseract

flist = ["test1.png", "test2.png"]

if len(sys.argv) > 1:
    flist = sys.argv[1:]

def ocr(fname):
    start_time = time.time()
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 11")
    stop_time = time.time()

    word_count = len(text.split())
    print(f'Word count: {word_count} took {(stop_time - start_time) * 1000} ms')

for fname in flist:
    ocr(fname)

