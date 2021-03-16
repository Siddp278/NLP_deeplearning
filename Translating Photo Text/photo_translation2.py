# import the necessary packages
# run -> python photo_translation2.py --image Images/jap_banner.png --lang jpn
# for simple words written, it handles the job well but if the background has image it isn't able to handle the job.
# if the image has deformed or graffiti like words it isn't able to handle processing well.
from textblob import TextBlob
import textblob.exceptions
import pytesseract # works for png.
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
ap.add_argument("-l", "--lang", required=True, help="language that Tesseract will use when OCR'ing")
ap.add_argument("-t", "--to", type=str, default="en", help="language that we'll be translating to")
ap.add_argument("-p", "--psm", type=int, default=13, help="Tesseract PSM mode")
args = vars(ap.parse_args())

"""
--image: The path to the input image to be OCR’d.
--lang: The native language that Tesseract will use when ORC’ing the image. Just go to tessdata folder to see all the 
        available languages it support.
--to: The language into which we will be translating the native OCR text.
--psm(imp): The page segmentation mode for Tesseract. Our default is for a page segmentation mode of 13, which treats the 
       image as a single line of text. For our last example today, we will OCR a full block of text of German. For this
       full block, we will use a page segmentation mode of 3 which is fully automatic page segmentation without Orientation 
       and Script Detection (OSD).
"""

# load the input image and convert it from BGR to RGB channel
# ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Pillow has the order RBG but OpenCV has the order BGR.

# OCR the image, supplying the country code as the language parameter
options = "-l {} --psm {}".format(args["lang"], args["psm"])
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
text = pytesseract.image_to_string(rgb, config=options)

# show the original OCR'd text
print("ORIGINAL")
print(text)

# translate the text into a different language
# tb = TextBlob(text)
# txt = tb.translate(to=args["to"])

txt = " "


try:
    tb = TextBlob(text)
    txt = tb.translate(to=args["to"])
except textblob.exceptions.TranslatorError:
    pass
except textblob.exceptions.NotTranslated:
    pass


# show the translated text
print("TRANSLATED")
print(txt)