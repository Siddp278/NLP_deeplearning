# Not really good, can't recognize words from images if they are even a little bit complex.
import pytesseract
# pip install googletrans==3.1.0a0
from googletrans import Translator


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
string = pytesseract.image_to_string(r'Images\japanese-word.png', lang='jpn')
# string = pytesseract.image_to_string(r'Images\easy.png',)
# print(string)
# print(type(str(string))) - can be converted into str class.
# print(pytesseract.image_to_data('handwriting photo.jpg'))

tra = Translator(service_urls=['translate.googleapis.com'])
result = tra.translate(str(string))
print(result.src)
print(result.dest)
print(result.text)