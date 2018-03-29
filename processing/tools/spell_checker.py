#Add these lines:
import nltk
#nltk.download('wordnet')
#nltk.download('brown')
#nltk.download('words')

from nltk.corpus import wordnet as WN
from nltk.corpus import stopwords
import re

if False:
    import enchant
    dictionary = enchant.Dict("en_US")


stop_words_en = set(stopwords.words('english'))

# Brown Wordlist
from nltk.corpus import brown
from nltk.corpus import words
import operator

extra_list = ["doth", "tis", "'d", "'s", "'t"]

word_set = set([word.lower() for word in set(words.words() + brown.words()+stopwords.words('english') + extra_list) ])



def tokens(sent):
    return nltk.word_tokenize(sent)

def SpellChecker(line, use_enchant = False, verbose=False):
    token_list = tokens(line)
    total_words = len(token_list)
    misspelled_words = 0
    misspelled_dict = {}
    for i in token_list:
        strip = i.strip()
        if use_enchant:
            if not dictionary.check(strip):
                misspelled_words += 1
        else:
            if not strip in word_set:
                misspelled_words += 1
                add_item(strip, misspelled_dict)
                
        if False:
            if not WN.synsets(strip) and strip not in stop_words_en:
                misspelled_words += 1
    if verbose:
        print_dict(misspelled_dict)
    return misspelled_words/total_words

def add_item(word, d):
    if word in d.keys():
        d[word] += 1
    else:
        d[word] = 1

def print_dict(d):
    sorted_x = sorted(d.items(), key=operator.itemgetter(1))
    for item in sorted_x:
        print(item)


def count_misspellings(file_path, verbose = False):
    regex = re.compile('[^a-zA-Z \n\'\-\t]')
    
    with open(file_path, "r") as f:
        text = f.read().lower()
        text = regex.sub('', text)
        text = text.replace("-", " ")
        return SpellChecker(text, verbose = verbose)

if __name__ == "__main__":
    path = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\35 Sonnets.txt"
    path = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\A Day with Lord Byron.txt"
    x = count_misspellings(path, verbose= True)
    print(x)
