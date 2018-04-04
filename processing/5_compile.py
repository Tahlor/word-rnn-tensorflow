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

DIR = r"../data/gutenberg"
TEST = False

# SUFFIX for files too big for github
# Files ending with LARGE are ignored

if TEST:
    OUTPUT = r"../data/test.txt"
    VOCAB = 300
    SUFFIX = ""    
else:
    OUTPUT = r"../data/large/gutenberg.txt"
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
    text = text.replace("\n", " |||| ")
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

        
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"-", " - ", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    # Remove tables of contents
    string = re.sub(r"[0-9]$", " @@ ", string)

    string = re.sub("[0-9]+$", " ", string)
    

    return string.strip()
        
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
    #in_pickle = OUTPUT+".pickle" + SUFFIX
    #text = pickle.load(open(in_pickle, "rb"))

# Should entire lines with uncommon words be deleted??
print("Pruning text...")
text = clean_str(text)
text = vocab_prune(text, VOCAB)
text = remove_lines(text)
out = OUTPUT.replace(".txt", "_restricted.txt" + SUFFIX)

# Output vocabulary restricted version
print("Writing text...")
pickle.dump(text, open(out+".pickle"+ SUFFIX, "wb"))
write_out(text, out)
