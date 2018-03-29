from langdetect import detect
import os
from tools.spell_checker import *

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Expects full path and destination folder
def move_on_language(ff, dest):
    f = os.path.basename(ff)
    try:
        with open(ff, "r") as fobj:
            lang = detect(fobj.read())
        if lang != "en":
            print("Removing {} \n Language {}".format( f, lang))
            os.rename(ff, os.path.join(dest, f))
    except:
        pass

# Expects full path and destination folder
def move_too_short(ff, dest, min_lines=400):
    f = os.path.basename(ff)
    with open(ff, "rb") as fobj:
        for i, l in enumerate(fobj):
            pass
        i += 1

    if i<min_lines:
        print("Removing {}, only {} lines".format(f, i))
        os.rename(ff, os.path.join(dest, f))

def move_misspellings(ff, dest, threshold = .05):
    if threshold > 1:
        threshold = threshold/100
    try:
        f = os.path.basename(ff)
        msp = count_misspellings(ff)

        if msp > threshold:
            print("File: {}, {}%".format(f, round(msp*100,2))) 
                os.rename(ff, os.path.join(dest, f))
    except:
        pass


DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
#DIR = r"D:\PyCharm Projects\~archive\BACKUP\data\gutenberg"
BAD_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\Failed\Not English"
SHORT_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\Failed\Short"
MISS_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\not clean"
              
for root, sub, files in os.walk(DIR):

    # Don't check subdirectories right now
    if root != DIR:
        continue

    for f in files:
        ff = os.path.join(root, f)
        #move_on_language(ff, BAD_DIR)
        #move_too_short(ff, SHORT_DIR, min_lines = 400)
        move_misspellings(ff, MISS_DIR)        
