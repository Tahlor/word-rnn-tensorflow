import os, os.path as path
import subprocess

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"

if path.isdir(DIR):
    for (root, dirs, files) in os.walk(DIR):
        for f in files:
            f = path.join(root, f)
            try:
                with open(f, mode='rt', encoding='utf-8') as input:
                    text = input.read()
            except (FileNotFoundError):
                print("File not found:({})".format(f))
                p = subprocess.Popen(["del", "/Q", "/S", '\\\\?\\'+f], cwd="D:", shell=True)
                continue

