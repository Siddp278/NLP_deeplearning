from langdetect import detect
import speech_recognition as sr

print(detect("War doesn't show who's right, just who's left."))

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")

print(audio_text)