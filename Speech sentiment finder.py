# To use this I have to install pyaudio - download the correct whl file, see pyaudio installation errors.
# Then run the code pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl and it will install.
# Next I installed python-docx


# Speech to Text

import speech_recognition as sr
from nltk.tokenize import TweetTokenizer
import  pickle, re, string
from nltk.corpus import stopwords, twitter_samples
import nltk

def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    # Time extension is of 1.5 minutes.
    try:
        # using google speech recognition
        data = r.recognize_google(audio_text)
        print("Text: " + data)
        print()
        # print("Text: " + r.recognize_google(audio_text), language = 'ta-IN') text is in tamil.
    except:
        print("Sorry, I did not get that")


"""
mydocx = docx.Document()
mydocx.add_paragraph(data)
mydocx.save("Audio_saved.docx")

"""
# Call over the pickled file for logprior and loglikelihood.
loaded_model1 = pickle.load(open('Doc_loglikelihood.pickle', 'rb'))
loaded_model2 = pickle.load(open('Doc_logptiot.pickle', 'rb'))

def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    """
    word_l = process_tweet(tweet)
    p = 0
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood.get(word)

    return p


my_tweet = data
logprior = int(loaded_model2)
loglikelihood = dict(loaded_model1)
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
if p > 0:
    print('The expression is positive')
if p == 0:
    print('The expression is neutral')
if p < 0:
    print('The expression is negative')



