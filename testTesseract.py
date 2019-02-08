import cv2
import sys
import pytesseract

if __name__ == '__main__':

  # if len(sys.argv) < 2:
  #   print('Usage: python ocr_simple.py image.jpg')
  #   sys.exit(1)

  # Read image path from command line
  # imPath = sys.argv[1]
  imPath = "abc.jpg"
  # Uncomment the line below to provide path to tesseract manually
  # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

  # Define config parameters.
  # '-l eng'  for using the English language
  # '--oem 1' for using LSTM OCR Engine
  config = ('-l eng --oem 1 --psm 7')

  # Read image from disk
  im = cv2.imread(imPath, cv2.IMREAD_COLOR)

  # Run tesseract OCR on image
  text = pytesseract.image_to_string(im, config=config)

  # Print recognized text
  print(text)


# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05
# python text_recognition.py --east frozen_east_text_detection.pb--image images/example_04.jpg
