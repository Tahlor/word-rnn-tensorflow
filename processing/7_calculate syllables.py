#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append(r"..")
import poetrytools
import os
import nltk

#D:\PyCharm Projects\word-rnn-tensorflow\meter

DIR = r"../data/gutenberg"
TEST = True

# SUFFIX for files too big for github
# Files ending with LARGE are ignored

if TEST:
    OUTPUT = r"../data/test/input.txt"
    VOCAB = 300
    SUFFIX = ""    
else:
    OUTPUT = r"../data/large/gutenberg.txt"
    VOCAB = 40000
    SUFFIX = "NEW"    

def write_out(text, f):        
    # Write out
    with open(f, "w") as fobj:
        fobj.write(text)


def read_in(ff):
    with open(ff, "r") as fobj:
        return fobj.read()


# Prune vocab for most common words
text = read_in(OUTPUT)

#write_out(text, out)
poem = poetrytools.tokenize(text) # need to tokenize the poem first
print(poem)
x = poetrytools.scanscion(poem)
print(x)

poetrytools.guess_form(poem, verbose=True)
