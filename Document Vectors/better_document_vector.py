# The model is to see if the two text written is written by the same author or not.
# Here to use GLOVE we have to convert the word embeddings from txt to word2vec format.
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk, sklearn, pickle, re, string
from nltk.corpus import stopwords
import numpy as np

def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


# load the Stanford GloVe model
# filename = 'glove.6B.300d.txt.word2vec'
# model = KeyedVectors.load_word2vec_format(filename, binary=False)
# save_data = open(r"C:\Users\KIIT\Desktop\Natural Language Processing\Pickle\word_embeddings_stanford.pickle", "wb")
# pickle.dump(model, save_data)
# save_data.close()

open_file = open(r"C:\Users\KIIT\Desktop\Natural Language Processing\Pickle\word_embeddings_stanford.pickle", "rb")
word_embeddings_data = pickle.load(open_file)
open_file.close()

# calculate: (king - man) + woman = ?
result = word_embeddings_data.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

# This was done for the author Shakespeare
f_var = open(r"C:\Users\KIIT\Desktop\Natural Language Processing\Shakespeare.txt", 'r')
text_list = []
# cleaned the text to get simple sentences.
for line in f_var:
    fields = line.split(".")
    for i in fields:
        if len(i) > 1:
            tweet = process_tweet(i)
            text_list.extend(tweet)
        else:
            pass

f_var.close()

# building a list to get individual words
word_list = []
for l1 in text_list:
    for item in l1:
        word_list.append(item.lower())

count = 0

document_embeddings = np.zeros(300)
for word in word_list:
    try:
        document_embeddings += word_embeddings_data[word]

    except:
        count = count + 1

print(len(word_list))
print(count) # This the number of words that werent recognized


# This was done for the author Christopher Marlowe
f_var1 = open(r"C:\Users\KIIT\Desktop\Natural Language Processing\Shakespeare2.txt", 'r')
text_list1 = []
# cleaned the text to get simple sentences.
for line in f_var1:
    fields1 = line.split(".")
    for i in fields1:
        if len(i) > 1:
            tweet1 = process_tweet(i)
            text_list1.extend(tweet1)
        else:
            pass

f_var1.close()

# building a list to get individual words
word_list1 = []
for l1 in text_list1:
    for item in l1:
        word_list1.append(item.lower())

count1 = 0

document_embeddings1 = np.zeros(300)
for word in word_list1:
    try:
        document_embeddings1 += word_embeddings_data[word]

    except:
        count1 = count1 + 1

print(len(word_list1))
print(count1) # This the number of words that werent recognized


# We will go for cosine_similarity
def cosine_similarity(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    """
    dot = np.dot(A, B)
    norma = np.sqrt(np.dot(A, A))
    normb = np.sqrt(np.dot(B, B))
    cos = dot / (norma * normb)

    return cos


similarity = cosine_similarity(document_embeddings, document_embeddings1)
print(similarity)
