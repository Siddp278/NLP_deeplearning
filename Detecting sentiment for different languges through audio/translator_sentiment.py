import speech_recognition as sr
from nltk.tokenize import TweetTokenizer
import pickle, re, string
from nltk.corpus import stopwords
import nltk
from googletrans import Translator, LANGUAGES



def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    # tweet = re.sub(r'^RT[\s]+', '', tweet)
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

def record(lang):
    # Reading Microphone as source
    # listening the speech and store in audio_text variable
    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source)
        print("Time over, thanks")
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # can I detect the language?
        if lang == 'en':
            said = r.recognize_google(audio_text, language='en-US')
        elif lang == 'es':
            said = r.recognize_google(audio_text, language="es")
        elif lang == 'ar-IQ': # Arabic
            said = r.recognize_google(audio_text, language="ar-IQ")
        elif lang == 'ja-JP': # Japanese
            said = r.recognize_google(audio_text, language="ja-JP")
        elif lang == 'es': # Spanish
            said = r.recognize_google(audio_text, language="es")
        else:
            said = r.recognize_google(audio_text, language="hi-IN")
    except:
        print("Some error occcured.")

    print("Text: " + said)
    print()
    # print("Text: " + r.recognize_google(audio_text), language = 'ta-IN') text is in tamil.
    return said


def TranslateFunc(input_text):
    lang = list(LANGUAGES.values())
    try:
        tra = Translator(service_urls=['translate.googleapis.com'])
        result = tra.translate(str(input_text))
        print(result.src)
        return result.text
    except:
        print("Sorry This Language is not available to translate")
        return None


data1 = 'アイヴァンモリスの翻訳のこの抜粋で、セイは宮廷生活のドラマを垣間見ることができます。彼女は、猫の命婦夫人に対する悲劇的な誤解が、法廷でお気に入りの犬であるオキナマロをどのように支持から失ったかについて語っています。花輪と花で覆われた後、彼は殴打されて追放され、この運命のねじれを元に戻して好意を取り戻すのに苦労しました。'

# Getting real time data, note auto detection of language is not done, so we have to input the language code.
translated_data = TranslateFunc(record('hi-IN'))


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


my_tweet = translated_data
logprior = int(loaded_model2)
loglikelihood = dict(loaded_model1)
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)

if p > 0:
    print('The expression is positive')
if p == 0:
    print('The expression is neutral')
if p < 0:
    print('The expression is negative')
