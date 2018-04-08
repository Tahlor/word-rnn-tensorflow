# -*- coding: utf-8 -*-
import os
import codecs
import collections
import numpy as np
import re
import itertools
from gensim.summarization.summarizer import summarize
import poetrytools
import sys

if sys.version_info[0] < 3:
    from six.moves import cPickle as pickle
else:
    import pickle


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding=None, simple_vocab = True, load_from_file = True):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.simple_vocab = simple_vocab

        if not os.path.isdir(data_dir):
            input_file = data_dir
            self.data_dir = os.path.abspath(os.path.join(input_file, os.pardir))
        else:
            input_file = os.path.join(data_dir, "input.txt")
        
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        self.save_file = os.path.join(data_dir, "batches.pickle")

        if load_from_file == True and os.path.exists(self.save_file):
            self.load_preprocessed(vocab_file, tensor_file)
            self.load_inputs()
        else:
            # Let's not read vocab and data from file. We many change them.
            if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
                print("reading text file")
                self.preprocess(input_file, vocab_file, tensor_file, encoding)
            else:
                print("loading preprocessed files")
                self.load_preprocessed(vocab_file, tensor_file)
            self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"\n\s*", r"  |||  ", string)

        # Separate symbols from words
        string = re.sub(r"([,.-;:!?~])", r" \1 ", string)
        string = re.sub(r"~", "--", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """

        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {}
        for i, x in enumerate(vocabulary_inv):
            vocabulary[x] = i
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        pickle_path = input_file.replace(".txt", ".pickle")
        use_pickle = False
        failed = True
        if os.path.exists(pickle_path) and use_pickle:
            try:
                with open(pickle_path, "rb") as pkl:
                    data = pickle.load(pkl)
                    failed = False
            except:
                pass

        if failed:
            with codecs.open(input_file, "r", encoding=encoding) as f:
                data = f.read()
            # Dump to pickle
            if use_pickle:
                with open(pickle_path, "wb") as fobj:
                    pickle.dump(data,fobj, 2)

        # Optional text cleaning or make them lower case, etc.
        if self.simple_vocab:
            data = self.clean_str(data)        

        # This is not optimized, shouldn't be too bad though
        x_text = ["\n" if i == "|||" else i for i in data.split()]
        #x_text = data.split()
        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.words, f,2)
        
        """print(data[0:500])
        # print(data1[0:500])
        print(x_text[0:500])
        print("\n" in self.vocab)
        print([x for x in self.vocab if "\n" in x])
        Stop
        """

        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = pickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        # Find end of line mark index
        #print(self.vocab)
        #print(self.words)

        self.endline_idx = self.vocab['\n']

        # Find synonym of endline words in corpus

        # Create sequence of vocab indices (inputs, outputs)
        # Giant list - [Steps in Epoch, Batch size, Sequence size]

        print("Batching it up...")
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

        self.get_last_words2(xdata)
        self.syllables = np.split(self.syllables.reshape(self.batch_size, -1), self.num_batches, 1)
        self.last_words = np.split(self.last_words.reshape(self.batch_size, -1), self.num_batches, 1)



        print("Getting last words/syllables")
        self.save_inputs()

    def save_inputs(self):
        d = {"x_batches":self.x_batches, "y_batches":self.y_batches, "last_words":self.last_words, "syllables":self.syllables}
        with open(self.save_file, 'wb') as f:
            pickle.dump(d, f,2)

    def load_inputs(self):
        with open(self.save_file, 'rb') as f:
            d = pickle.load(f)
        self.x_batches  = d["x_batches"]
        self.y_batches  = d["y_batches"]
        self.last_words = d["last_words"]
        self.syllables  = d["syllables"]
        self.endline_idx = self.vocab['\n']



    def tokenize_by_line(self, input_list):
        temp_list = np.copy(self.input_list)
        for batch in temp_list:
            for element in batch:
                last_word = element[::-1][0]
                next_line = []
                all_lines = []
                for n, word in enumerate(element[::-1]):
                    if word == self.endline_idx and n + 1 < len(element):
                        all_lines.append(next_line)
                        next_line = []
                    next_line.append(word)
                    del element[::-1][n]
                element = all_lines

    def get_syllables(self):
        import poetrytools
        self.syllables = poetrytools.scanscion(self.x_batches)
        for line in self.syllables:
            pass

    def get_last_words2(self, xdata):
        self.last_words = np.copy(xdata)
        self.syllables = np.copy(xdata)

        line_start = 0
        last_word = self.last_words[-1]
        l = xdata[::-1].reshape(-1)

        #m = xdata[-1]
        #print(m)
        #print(self.words[m])
        for n, word in enumerate(l):
            #print(self.words[word])
            if word == self.endline_idx or n + 1 == len(l):
                line_slice = slice(line_start,n+1)
                line = l[line_slice]
                #print(list(np.asarray(self.words)[line]))
                syllables = self.get_syllables_from_indices(line)

                # Make replacement
                self.syllables[::-1][line_slice] = syllables
                self.last_words[::-1][line_slice] = last_word
                last_word = self.get_next_word(l, n+1)
                line_start = n+1
        #print(list(np.asarray(self.words)[self.last_words[0:100]]))
        #print(list(np.asarray(self.words)[xdata[0:100]])

        np.set_printoptions(formatter={'int_kind': lambda x:  "{:0>3d}".format(x)})

        print(self.syllables[0:100])
        print(self.last_words[0:100])
        print(xdata[0:100])




    def get_next_word(self, array, idx):
        for n, word_idx in enumerate(array[idx:]):
            # Return first thing longer than 1 character
            word = self.words[word_idx]
            if len(word)>1 or word.isalpha():
                #print(self.words[word_idx])
                return word_idx

    def get_syllables_from_indices(self, indices):
        words = [x for x in list(np.asarray(self.words)[indices]) if len(x)>1 or x.isalpha()]
        syllables = len(''.join([poetrytools.stress(x, "min") for x in words]))
        return syllables

    def get_non_symbol(self, l):
        for el in l:
            if self.words[el].isalpha():
                return el

        # If not found return empty line
        return self.endline_idx

    #X is input, Y is output
    def next_batch(self, dropout = .2): # dropout = 0 means no dropout
        x, y, last_words, syllables = self.x_batches[self.pointer], self.y_batches[self.pointer], self.last_words[self.pointer], self.syllables[self.pointer]
        self.pointer += 1
        if dropout > 0:
            dropout_mask = np.random.binomial([np.ones(last_words.shape[:-1])], 1 - dropout)[0][..., None]
            dropout_mask2 = np.random.binomial([np.ones(syllables.shape[:-1])], 1 - dropout)[0][..., None]

            if self.endline_idx == 0:
                last_words = last_words * dropout_mask
            else: # if endline is not the first vocab word
                last_words = last_words * dropout_mask + (1-dropout_mask) * self.endline_idx

            # Syllable dropout - 0
            syllables = syllables * dropout_mask2

        return x, y, last_words, syllables

    def reset_batch_pointer(self):
        self.pointer = 0

    def get_last_word():
        pass
        
    def get_sentiment():
        pass

    def get_summary():
        gensim.summarization.summarizer.summarize(text, ratio=0.2, word_count=None, split=False)

if __name__ == "__main__":
    path = r"./data/test"
    #path = r"./data/FINAL"

    # ARGS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=path,
                        help='include extra input features')

    args = parser.parse_args()

    data_loader = TextLoader(args.path, 10, 50, load_from_file=False)
    x,y,z,syl = data_loader.next_batch()
    #print(data_loader.words[x[0].astype(int)])
    print(syl[0:10])

    print("Lsst words")
    print(z[0:10])

    print(data_loader.words[syl[0:10]])