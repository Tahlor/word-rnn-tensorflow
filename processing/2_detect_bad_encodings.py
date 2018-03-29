import os
import re

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
BAD_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\failed"

for root, sub, files in os.walk(DIR):
    # Only do top level
    if root != DIR:
        continue

    for f in files:
        ff = os.path.join(root, f)
        try:
            with open(ff, "r") as fobj:
                fobj.read()
        except:
            print("Moving {} to failed".format(f))
            os.rename(ff, os.path.join(BAD_DIR, f))

                        
