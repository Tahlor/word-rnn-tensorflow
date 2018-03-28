import argparse
import os, os.path as path
import subprocess
import chardet
import sys

"""
parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory in which to convert file encodings.')
parser.add_argument('-e', '--newencoding', default='utf-8', help='"Destination" encoding. When the script finishes, all files in the ' +
                                                'specified directory should have this encoding.')
args = parser.parse_args()
"""

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
NEW_ENCODING = "utf-8"


other = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\A Defence of Poesie and Poems.txt"
x = open(other, "rt", encoding="utf-8")

#codecs.open(file_name, "r",encoding='utf-8', errors='ignore') 

#if path.isdir(args.directory):
if path.isdir(DIR):
    # objs = os.listdir(args.directory)
    # files = [path.join(args.directory, f) for f in objs if not path.isdir(f)]
    for (root, dirs, files) in os.walk(DIR):
        for f in files:
            f = path.join(root, f)

            # If no problems with new encoding, continue
            try:
                with open(f, mode='rt', encoding=NEW_ENCODING) as i:
                    i.read()
            except Exception as e:
                print("Failed {}".format(f))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_type)
                input("OK")
                try:
                    # Read byte format
                    with open(f, mode='rb') as i:
                        guessed_encoding = chardet.detect(i.read())

                    # Read as encoding
                    with open(f, mode='rt', encoding=guessed_encoding["encoding"]) as i:
                        text = i.read()

                except:
                    print('Encoding does not match expected encoding...' + f)
                    print("Guess: {}".format(guessed_encoding))
                    continue

                # Convert to New Encoding
                if guessed_encoding != NEW_ENCODING:
                    print('Converting...' + f)
                    print('Encoding...' + guessed_encoding["encoding"])
                    with open(f, mode='wt', encoding=NEW_ENCODING) as output:
                        output.write(text)


else:
    raise 'Invalid directory: ' + args.directory
