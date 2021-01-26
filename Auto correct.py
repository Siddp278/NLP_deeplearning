import re
from collections import Counter


def process_tweet(tweet):
    """
    :param tweet: a string that is to be processed i.e. cleaned
    :return: cleaned string.
    """
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    return tweet


def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []

    with open(file_name) as f:
        file_name_data = f.read()
    file_name_data = process_tweet(file_name_data.lower())
    words = re.findall(r'\w+', file_name_data)

    return words


word_l = process_data('Shakespeare.txt')
vocab = set(word_l)
print(f"There are {len(vocab)} unique words in the vocabulary.")

"""
Implement a get_count function that returns a dictionary

The dictionary's keys are words
The value for each word is the number of times that word appears in the corpus.
"""


def get_count(word_l):
    """
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    """

    word_count_dict = {}
    word_count_dict = Counter(word_l)

    return word_count_dict


word_count_dict = get_count(word_l)
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")

"""
Given the dictionary of word counts, compute the probability that each word
will appear if randomly selected from the corpus of words.
P(w_i) = C(w_i) \M; M=sum of all occurrances of words, C(w_i)=sum of occurrances of w_i word.
"""


def get_probs(word_count_dict):
    """
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    """

    probs = {}  # return this variable correctly
    m = sum(word_count_dict.values())
    for key in word_count_dict.keys():
        probs[key] = word_count_dict.get(key, 0) / m

    return probs


probs = get_probs(word_count_dict)
print(f"P('thee') is {probs['thee']:.4f}")


"""
Part 2: String Manipulations
Now, that you have computed P(w_i) for all the words in the corpus, you will write a few functions to manipulate strings so that you can edit the erroneous strings and return the right spellings of the words. In this section, you will implement four functions:

delete_letter: given a word, it returns all the possible strings that have one character removed.
switch_letter: given a word, it returns all the possible strings that have two adjacent letters switched.
replace_letter: given a word, it returns all the possible strings that have one character replaced by another different letter.
insert_letter: given a word, it returns all the possible strings that have an additional character inserted.

"""


# delete_letter()
def delete_letter(word, verbose=False):
    """
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    """
    delete_l = []
    split_l = []

    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    delete_l = [L + R[1:] for L, R in split_l if R]

    if verbose:
        print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")# printing implicitly.

    return delete_l


# checking the function
delete_word_l = delete_letter(word="cans", verbose=True)


# switch_letter()
def switch_letter(word, verbose=False):
    """
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    """
    def swap(c, i, j):
        c = list(c)
        c[i], c[j] = c[j], c[i]
        return ''.join(c)

    switch_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    switch_l = [a + b[1] + b[0] + b[2:] for a, b in split_l if len(b) >= 2]

    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


# testing the function
switch_word_l = switch_letter(word="eta", verbose=True)


# replace_letter()
def replace_letter(word, verbose=False):
    """
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    """

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []

    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l = [a + l + (b[1:] if len(b) > 1 else '') for a, b in split_l if b for l in letters]
    replace_set = set(replace_l)
    replace_set.remove(word)
    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose:
        print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


# testing the functions
replace_l = replace_letter(word='can', verbose=True)


#  insert_letter()
def insert_letter(word, verbose=False):
    """
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    insert_l = [ a + l + b for a, b in split_l for l in letters]

    if verbose:
        print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


# testing the function
insert_l = insert_letter(word='at', verbose=False)

"""
Combining the edits:
Now that you have implemented the string manipulations, you will create two functions that,
 given a string, will return all the possible single and double edits on that string. These will
 be edit_one_letter() and edit_two_letters().
"""


#  Edit one letter
def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    if allow_switches:
        edit_one_set.update(switch_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))

    return edit_one_set


# Edit two letters
def edit_two_letters(word, allow_switches=True):
    """
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    """

    edit_two_set = set()
    edit_one = edit_one_letter(word, allow_switches=allow_switches)
    for w in edit_one:
        if w:
            edit_two = edit_one_letter(w, allow_switches=allow_switches)
            edit_two_set.update(edit_two)

    return edit_two_set


# suggest spelling suggestions
def get_corrections(word, probs, vocab, n=2, verbose=False):
    """
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    """

    suggestions = []
    n_best = []
    suggestions = list((word in vocab and word) or edit_one_letter(word).intersection(vocab) or
                       edit_two_letters(word).intersection(vocab))
    n_best = [[s, probs[s]] for s in list(reversed(suggestions))]

    if verbose:
        print("suggestions = ", suggestions)

    return n_best


# Test your implementation
my_word = 'bob'
tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=False)
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

# print(f"data type of corrections {type(tmp_corrections)}")
