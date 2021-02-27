# NLP_deeplearning

## Some common preprocessing steps for the language models include:
### **Audio_sentiment**
We are using `Speech Recognizer` Package to convert the audio clipping to text, from there on we are using naive bayes model(predicting the most probable word-sentiment) to train that model for sentiment analysis on that text. The `Sentiment_analysis_naive` has the implementation of the naive bayes model. Have included the `pickle` files so that you dont have to train the model again and again.

### - **Autocorrect System**
It has the implementation of autocorrect system using the `edit distance` algorithm with 4 operations that are **replace, switch, delete and insert**. We input those operations on our incorrect word and then find the most probable among the updated words to see which one fits the bill using probabilty of that word occurring in the corpus. **NOTE**: a text data (Shakespeare.txt) is provided as corpus.

### - **Detecting sentiment for different languges through audio**
This is just the upgraded version of Audio sentiment where it supports multiple languages meaning the audio can be in any language and we can find the sentiment for that audio. **NOTE**: The `translator_sentiment.py` is the main driver file and `PyAudio` application file is provided as for some systems the normal package installed through pip may not work.

### - **Document Vectors**
`Word Embeddings` or `Word vectorization` is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which used to find word predictions, word similarities/semantics. An extension of this is `Document Vectorization` meaning it vectorizes the whole document text to fit into numbers(vector). Using this I have implemented finding similarity between two text(documents). I have taken two author's works and tried to find how similar/disimilar they are(data/works is provided).
