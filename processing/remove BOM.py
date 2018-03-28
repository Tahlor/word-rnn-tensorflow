import argparse
import os, os.path as path
import subprocess
import chardet
import sys

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
NEW_ENCODING = "utf-8"

other = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\A Defence of Poesie and Poems.txt"
x = open(other, "rt", encoding="utf-8")

if path.isdir(DIR):
    for (root, dirs, files) in os.walk(DIR):
        for f in files:
            f = os.path.join(root,f)
            with open(f, mode='rb+') as i:
                
                i.write(i.read()[3:])
