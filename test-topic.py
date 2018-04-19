import os
import sys
from six.moves import cPickle
from nltk import tokenize

def get_topic(poem):
    indices = [vocab[w] for w in poem]
    return words[max(indices)]

poem = r"""As it fell upon a day 
In the merry month of May, 
Sitting in a pleasant shade 
Which a grove of myrtles made, 
Beasts did leap and birds did sing, 
Trees did grow and plants did spring, 
Every thing did banish moan 
""".lower()

save = r"./save/MASTER"
with open(os.path.join(save, 'words_vocab.pkl'), 'rb') as f:
    if sys.version_info[0] >= 3:
        words, vocab = cPickle.load(f, encoding='latin-1')
    else:
        words, vocab = cPickle.load(f)


x = tokenize.word_tokenize(poem)
print(get_topic(x))
