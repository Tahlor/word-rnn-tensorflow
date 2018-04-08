#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import re
import nltk
#nltk.download('perluniprops')

from nltk.tokenize.moses import MosesDetokenizer
from collections import Counter
import heapq
import operator
from operator import *
import pickle
from unidecode import unidecode

DIR = r"../data/gutenberg"
TEST = False
#TEST = True

newline = "|||||"
# SUFFIX for files too big for github
# Files ending with LARGE are ignored

if TEST:
    OUTPUT = r"../data/test/input2.txt"
    VOCAB = 3000
    SUFFIX = ""    
else:
    OUTPUT = r"../data/FINAL/gutenberg2.txt"
    VOCAB = 40000
    SUFFIX = "NEW"    

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

def vocab_prune(text, keep_words = 100000, remove_apostrophe = True):
    print("Tokenizing...")
    #text = text.replace("\n", " {} ".format(newline))
    token_list = nltk.word_tokenize(text.lower(), preserve_line = True)


    print("Finding least common words...")
    # Get word freq
    cnt = Counter()
    for word in token_list:
        cnt[word] += 1

    # Get least common
    print("Total words: {}".format(len(cnt)))
    eliminate = max(len(cnt) - keep_words,0)
    lc = least_common((cnt), eliminate)

    # Regex remove ridiculous apostrophes
    awords = []
    if remove_apostrophe:
        apostrophed_words = re.compile(".*\'[A-z][A-z][A-z]+")

        for word in token_list:        
            if apostrophed_words.match(word) != None:
                awords.append(word)
        awords = list(set(awords))
               
    print("Last words out: {}".format(lc[-10:]))    

    # Prep for devocabularization
    d = {"''":'"', "``":'"'}
    
    stop_words = ["deg", ]
    
    # Replace uncommon words with @@
    for word, freq in lc:
        d[word] = "@@"
    for word in awords + stop_words:
        d[word] = "@@"

    print("Deleting least common words...")
    # Replace items in list
    l = len(token_list)
    #print(token_list)
    start = 0
    end = 0
    found_bad = False
    delete_list = []
    for n, word in enumerate(token_list):
        if word == newline:
            if found_bad == False:
                start = n
            else:
                end = n
                found_bad = False
                delete_list.append(slice(start,end))
        if word in d:
            found_bad = True
        if n % 1000000 == 0:
            print(n*1.0/l)

    for item in delete_list[::-1]:
        del token_list[item]
    """for n, word in enumerate(token_list):
        if word in d.keys():
            token_list[n] = "@@"
        if n % 10000 == 0:
            print(n*1.0/l)"""


    #token_list = [d[word] if word in d.keys()  else word for word in token_list]

    print("Untokenizing...")
    text = MosesDetokenizer().detokenize(token_list, return_str=True)

    print("Restoring line characters")

    nl = re.compile("\s*{}+\s*".format(re.escape(newline[:-1])))
    text = nl.sub(r"\n", text)

    return text


def least_common(adict, n=None):
    if n is None:
        return sorted(adict.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(n, adict.items(), key=itemgetter(1))

def read_in(ff):
    import codecs
    with codecs.open(ff, "r", encoding = "utf-8", errors="ignore") as fobj:
        return fobj.read()

        
def clean_str(string):
    string = re.sub(r"—", "~", string)  # fix dashes
    string = re.sub(r"--", "~", string)  # fix dashes
    string = unidecode(string)
    string = re.sub(r"\s*\n\s*", " {} ".format(newline), string) # delete crazy characters
    string = re.sub(r"[^A-Za-z0-9,!\?\'\`\-\|\n\.:;~]", " ", string) # delete crazy characters

    # Only keep apostrophe's in the middle of words
    string = re.sub(r"([A-Za-z])(')([A-Za-z])", r"\1@@@\3", string)
    string = re.sub(r"s'", "s@@@", string)
    string = re.sub(r"'", " ", string)
    string = re.sub(r"@@@", "'", string)
    string = re.sub(r"\'s", " \'s", string)

    # Handle hypens
    string = re.sub(r"-", " - ", string)

    # Separate punctuation
    string = re.sub(r";", " ; ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", r" ! ", string)
    string = re.sub(r"\?", r" ? ", string)
    
    # Remove tables of contents - may also remove some lines with line numbers

    string = re.sub("(?m)^.*[0-9]+\s*$", "", string)
    string = re.sub("[0-9]+", " ", string) # remove all other numerics

    # Remove double punctuation
    string = re.sub(r'([~\-.?,:;])([~\-.?,:;\s]*)', r' \1 ', string)
    string = re.sub(r'~', r'--', string) # sub back in --
    string = re.sub(r"\s{2,}", " ", string)


    return string.strip().lower()
        
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
    write_out(text, OUTPUT + SUFFIX)
    pickle.dump(text, open(OUTPUT+".pickle" + SUFFIX, "wb"))

# Prune vocab for most common words
if TEST:
    text = read_in(OUTPUT + SUFFIX)
else:
    print("Loading pickle...")
    text = read_in(OUTPUT + SUFFIX)

# Should entire lines with uncommon words be deleted??
print("Pruning text...")
text = clean_str(text)
text = vocab_prune(text, VOCAB, remove_apostrophe = False)
text = remove_lines(text)
out = OUTPUT.replace(".txt", "_restricted.txt" + SUFFIX)

# Output vocabulary restricted version
print("Writing text...")
#pickle.dump(text, open(out+".pickle"+ SUFFIX, "wb"))
write_out(text, out)
