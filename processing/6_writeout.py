import os
import pickle

DIR = r"../data/gutenberg"
TEST = False
SUFFIX = "LARGE"
OUTPUT = r"../data/gutenberg.txt"

out = OUTPUT.replace(".txt", "_restricted.txt" + SUFFIX)

def write_out(text, f):        
    with open(f, "w", encoding = "utf-8") as fobj:
        fobj.write(text)

print("Loading pickle...")
text = pickle.load(open(out+".pickle" + SUFFIX, "rb"))
write_out(text, out)
