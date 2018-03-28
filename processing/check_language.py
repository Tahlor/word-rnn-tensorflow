from langdetect import detect
import os

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
BAD_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\Failed\Not English"
for root, sub, files in os.walk(DIR):
    if root != DIR:
        continue
    for f in files:
        ff = os.path.join(root, f)
        #print(ff)
        with open(ff, "r") as fobj:
            lang = detect(fobj.read())
        if lang != "en":
            print("Removing {} \n Language {}".format( f, lang))
            os.rename(ff, os.path.join(BAD_DIR, f))
            

