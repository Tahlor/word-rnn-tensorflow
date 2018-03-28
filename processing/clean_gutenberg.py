import os

# Input a list of things to find before first string or just first string
def parse(text, begin, end, exclude_targets = True):

    # Position 1
    pos1 = 0
    if begin == None:
        pos1 = 0
    elif type(begin) == type([]):
        for i in begin:
            pos1 += text[pos1:].find(i)
            pos1 += len(i)
    else:
        pos1 = text.find(begin)

    if not exclude_targets:
        pos1 -= len(i)

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
    if exclude_targets:
        pos1 = pos1 + len(begin)
    else:
        pos2 = pos2 - len(end)

    return text[pos1:pos2]

def clean_text(text):
    # Strip Gutenberg
    text = parse(text, ["*** START OF THIS PROJECT GUTENBERG", " ***"], "*** END OF THIS PROJECT GUTENBERG")
    #return text
    text_list = text.split("\n")
    out_list = []
    for line in text_list:
        line = line.strip()

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

        # Remove project gutenberg
        if "project gutenberg" in line.lower():
            continue

        # Remove internal digits? No, we'll ignore them in training
        #[i for i in s if not i.isdigit()]

        out_list.append(line)
    text = "\n".join(out_list)
    return text
    
    
DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
BAD_DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg\failed"

for root, sub, files in os.walk(DIR):
    for f in files:
        print(f)
        ff = os.path.join(root, f)
        try:
            with open(ff, "r") as fobj:
                text = clean_text(fobj.read())
            with open(ff, "w") as fobj:
                #print(text[0:10000])
                fobj.write(text)
        except:
            os.rename(ff, os.path.join(BAD_DIR, f))

                        
