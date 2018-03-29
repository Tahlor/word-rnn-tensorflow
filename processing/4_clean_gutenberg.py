import os
import re

# Input a list of things to find before first string or just first string
def parse(text, begin, end, exclude_targets = True, full_list = False):

    # Position 1
    # If full_list false specified, this presently will ignore elements not found
    pos1 = 0
    if begin == None:
        pos1 = 0
    elif type(begin) == type([]):
        for n, i in enumerate(begin):
            offset = text[pos1:].find(i)
            if offset > -1:
                pos1 += offset

                # Move completely past found term, except for last time
                if n + 1 < len(begin):
                    pos1 += len(i)
            elif full_list:
                pos1 = 0
                break
    else:
        pos1 = text.find(begin)
    
    # By default, start at beginning if not found
    if pos1 == -1:
        pos1 = 0

    # Position 2
    if end == None:
        pos2 = None
    else:
        pos2 = text[pos1:].find(end)

        # By default, go to end if no match is found
        if pos2 == -1:
            pos2 = None

    # Include search text
    if exclude_targets and pos1 > 0:
        pos1 += len(begin)
    elif not exclude_targets and not pos2 is None:
        pos2 += len(end)

    return text[pos1:pos2]

def clean_text(text):
    # Strip Gutenberg
    text = parse(text, ["*** START OF THIS PROJECT GUTENBERG", " ***"], "*** END OF THIS PROJECT GUTENBERG")
    #return text

    text_list = text.split("\n")
    out_list = []
    for line_number, line in enumerate(text_list):
        line = line.strip()

        #print(line)
        # Strip notes - lines that start with [
        if line == "" or line[0] == "[":
            continue

        # Strip Numerals
        if line[0].isdigit():
            continue

        # Roman Numerals

        # Strip ALL CAPS lines
        if line.isupper():
            continue

        # Ignore words
        ignore_at_beginning = ["produced by", "title: ", "author: ", "language: english",
              "character set encoding", "produced from ", "all rights reserved",
              "ltd.", "reprinted", "first edition"]

        ignore_anywhere = ["*", "<", ">", "[", "]"]
        ignore_anywhere = []
        
        stop_words = ["copyright", "gutenberg", "proofreading", "http:", "www.", "internet", "ebook", ".zip"]

        if (has_numerals(line) or has_stop_word(line, ignore_at_beginning)) and line_number < 100:
            continue

        if has_stop_word(line, ignore_anywhere):
            continue

        if re.match(".*[\*\[\]<>]", line):
            continue
        
        # Stop words
        if has_stop_word(line, stop_words):
            if line_number > 500:
                break
            else:
                continue

        out_list.append(line)
    text = "\n".join(out_list)
    #print(text[0:1000])
    return text


# Ignore this line only
def has_stop_word(line, stop_words):
    for word in stop_words:
        if line.lower().find(word) >-1:
            return True
    return False

def has_numerals(line):
    # Chapter/Roman Numerals
    if re.match("^[IXVL]+\. ", line):
        return True

    if re.match("[A-Z]+ [IXVL]+\. ", line):
        return True


    # E.g. Page numbers
    if re.match(".*[0-9]+$", line):
        return True

    
    return False
    
DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\test"
DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
BAD_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\failed"

for root, sub, files in os.walk(DIR):
    for f in files:

        # Only do top level
        if root != DIR:
            continue
        print(f)
        ff = os.path.join(root, f)
        try:
            with open(ff, "r") as fobj:
                text = clean_text(fobj.read())

            #print(text[0:500])
            #print(text[-500:])
            
            #input("Stop")      
            with open(ff, "w") as fobj:
                fobj.write(text)
        except:
            Stop
            input("Move {} to failed?".format(f))
            os.rename(ff, os.path.join(BAD_DIR, f))

                        
