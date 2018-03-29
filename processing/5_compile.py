import os
import re
import nltk
from nltk.tokenize.moses import MosesDetokenizer
from collections import Counter
import heapq
import operator
from operator import *
import pickle
#nltk.download('perluniprops')
    
DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
TEST = False
if TEST:
    OUTPUT = r"D:\PyCharm Projects\word-rnn-tensorflow\data\test.txt"
    VOCAB = 300
else:
    OUTPUT = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg.txt"
    VOCAB = 100000

def create_corpus(DIR, recursive = False):
    for root, sub, files in os.walk(DIR):
        # Only do top level
        if recursive and root != DIR:
            continue
        text = ""
        
        for f in files:
            print(f)
            ff = os.path.join(root, f)

            with open(ff, "r") as fobj:
                text += "\n\n\n" + fobj.read()
    return text

def write_out(text, f):        
    # Write out
    with open(f, "w") as fobj:
        fobj.write(text)

def vocab_prune(text, keep_words = 100000):
    print("Tokenizing...")
    text = text.replace("\n", " |||| ")
    token_list = nltk.word_tokenize(text, preserve_line = True)


    print("Finding least common words...")
    # Get word freq
    cnt = Counter()
    for word in token_list:
        cnt[word] += 1

    # Get least common
    print("Total words: {}".format(len(cnt)))
    eliminate = max(len(cnt) - keep_words,0)
    lc = least_common((cnt), eliminate)

    print("Last words out: {}".format(lc[-10:]))    

    # Prep for devocabularization
    d = {"''":'"', "``":'"'}
    for word, freq in lc:
        d[word] = "@@"

    print("Deleting least common words...")
    # Replace items in list
    token_list = [d[word] if word in d.keys()  else word for word in token_list]

    print("Untokenizing...")
    text = MosesDetokenizer().detokenize(token_list, return_str=True)

    print("Restoring line characters")

    nl = re.compile("\s*\|\|\|\|\s*")
    text = nl.sub(r"\n", text)

    return text


def least_common(adict, n=None):
    if n is None:
        return sorted(adict.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(n, adict.items(), key=itemgetter(1))

def read_in(ff):
    with open(ff, "r") as fobj:
        return fobj.read()

def remove_lines(text, char = "@@"):
    text_list = text.split("\n")

    out_list = []
    for line_number, line in enumerate(text_list):
        line = line.strip()
        if line.find(char) > -1:
            continue
        out_list.append(line)
    return "\n".join(out_list)
        

# Read in corpus
if False:
    text = create_corpus(DIR)
    write_out(text, OUTPUT)
    pickle.dump(text, open(OUTPUT+".pickle", "wb"))

# Prune vocab for most common words
if TEST:
    text = read_in(OUTPUT)
else:
    print("Loading pickle...")
    text = pickle.load(open(OUTPUT+".pickle", "rb"))


# Should entire lines with uncommon words be deleted??
print("Pruning text...")
text = vocab_prune(text, VOCAB)
text = remove_lines(text)
out = OUTPUT.replace(".txt", "_restricted.txt")

# Output vocabulary restricted version
print("Writing text...")
pickle.dump(text, open(out+".pickle", "wb"))
write_out(text, out)
