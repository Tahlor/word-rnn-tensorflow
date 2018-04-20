from __future__ import print_function
import numpy as np
import tensorflow as tf
import re
import language_check

import argparse
import os
from six.moves import cPickle

from model import Model
import sys
import datamuser
import random
import string


TAYLOR = False
try:
    if os.environ["COMPUTERNAME"] == 'DALAILAMA':
        TAYLOR = True
except:
    TAYLOR = False

# Sterling's globals
save_dir = 'save'

if TAYLOR:
    save_dir = r"./save/MASTER"
    #save_dir = r"./save/FINAL"

TOP_TOPIC_WORDS = 30
TOP_VOCAB_WORDS = 5000

class PoemWriter():

    def __init__(self, save_dir='save', n=50, prime = ' ', count = 1, end_word = "turtle", output_path = "sample.txt", internal_call = False, model = None, syllables = 10, pick = 1, use_topics = False):

        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', '-s', type=str, default=save_dir,
                            help='model directory to load stored checkpointed models from')
        parser.add_argument('-n', type=int, default=n,
                            help='number of words to sample')
        parser.add_argument('--prime', type=str, default=prime,
                            help='prime text')
        parser.add_argument('--pick', type=int, default=1,
                            help='1 = weighted pick, 2 = beam search pick')
        parser.add_argument('--width', type=int, default=5,
                            help='width of the beam search')
        parser.add_argument('--sample', type=int, default=1,
                            help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
        parser.add_argument('--count', '-c', type=int, default=count,
                            help='number of samples to print')
        parser.add_argument('--quiet', '-q', default=False, action='store_true',
                            help='suppress printing the prime text (default false)')
        parser.add_argument('--end_word', '-e', default=end_word,
                            help='Last word of line')
        parser.add_argument('--output_path', '-o', default=output_path,
                            help='Last word of line')
        parser.add_argument('--syllables', '-y', default=syllables,
                            help='Last word of line', type=int)
        parser.add_argument('--use_topics', '-t', default=use_topics,
                            help='Use topic words', type=bool)

        self.args = parser.parse_args("")

        path = os.path.join(self.args.save_dir, 'config.pkl')
        with open(path, 'rb') as f:
            saved_args = cPickle.load(f)
            saved_args.use_topics=self.args.use_topics

        main_path = os.path.join(self.args.save_dir, 'words_vocab.pkl')
        freq_path = os.path.join(self.args.save_dir, 'words_vocab_freq.pkl')

        self.words_freq, self.vocab_freq = self.open_pickle(freq_path)
        self.words, self.vocab = self.open_pickle(main_path)

        self.model = Model(saved_args, True)
        self.freq_words = set(self.words_freq[0:TOP_VOCAB_WORDS ])

    def open_pickle(self, path):
        with open(path, 'rb') as f:
            if sys.version_info[0] >= 3:
                words, vocab = cPickle.load(f, encoding='latin-1')
            else:
                words, vocab = cPickle.load(f)
        return words, vocab


    def evaluate_line(self, line):
        return True  # for now

    def strip_punc(self, s):
        table = str.maketrans({key: None for key in string.punctuation})
        new_s = s.translate(table)  # Output: string without punctuation
        return new_s

    def cap(self, match):
        return match.group().upper()

    def clean_poem(self, poem_list):
        print("cleaning poem")
        for i, l in enumerate(poem_list):
            l = l.strip()
            l = l.replace("- ", "-")
            l = l.replace(" i ", " I ")
            l = l[0].upper() + l[1:]
            l = re.sub("(\. [a-z])", self.cap, l)
            poem_list[i] = l
        return poem_list

    def correct_grammar(self, poem):
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(poem)
        for i in matches:
            print(matches[i])
        new_poem = language_check.correct(poem, matches)
        return new_poem


    def sample(self, num_syllables, num_lines, topic_word):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
            text_list = []

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                continuous = True
                while continuous:
                    # Get related words
                    related_words = datamuser.get_all_related_words(topic_word.split(), TOP_TOPIC_WORDS)
                    print(related_words)
                    print(self.vocab.keys())
                    topic_words = list(related_words.intersection(self.freq_words))
                    if len(topic_words) == 0:
                        raise ValueError("No vocab words related to topic")
                    print (topic_words)


                    # Do priming
                    prime = ""
                    if self.args.prime == "":
                        for ii in range(0, int(num_syllables/2)):
                            prime += random.choice(topic_words) + " "
                    else:
                        prime = self.args.prime

                    if prime[-1] != "\n":
                        prime += ".\n"

                    # prime = '{} {} {}\n'.format(topic_word, topic_word, topic_word)
                    quiet = True
                    poem_lines = []
                    print("Prime {}".format(prime))
                    i = 0
                    while i < num_lines:


                        # get endword

                        if i % 2 == 0:
                            # pick random topic word
                            end_word = random.choice(topic_words)
                            end_word = "\n"
                        else:
                            # rhyme with last line
                            rhymes = list(datamuser.get_rhymes(last_word, weak_rhymes=False).intersection(self.freq_words))
                            # print (len(rhymes))
                            if len(rhymes)==0: end_word = "\n" #random.choice(topic_words)
                            else: end_word = random.choice(rhymes)
                            # print('LAST WORD: {}  END WORD: {}'.format(last_word, end_word))

                        temp_topic_word = random.choice(topic_words)
                        print("Topic influencer {}".format(temp_topic_word))
                        # print ('END WORD: {}'.format(end_word))
                        # end_word = 'flag'

                        candidate_lines = []
                        scores = []
                        self.args.sample = 1
                        print ("GENERATING A NEW LINE -- SAMPLING SOME CANDIDATES -- END WORD {}".format(end_word))
                        for j in range(10):  # get best of 10 lines
                            line, score = self.model.sample(sess, self.words, self.vocab, self.args.n,
                                              prime, self.args.sample, self.args.pick,
                                              self.args.width, quiet, end_word, num_syllables, True, topic_word=temp_topic_word)
                            # quiet = True
                            # line = lines[len(prime):].split('\n')[0]  # strip off prime and keep next single line
                            candidate_lines.append(line)
                            scores.append(score)
                            # print ("LINE: {}  SCORE: {}".format(lines, score))

                        print(scores)
                        print(np.argmax(scores))
                        if scores[np.argmax(scores)] == -40:
                            print(candidate_lines)

                        line = candidate_lines[np.argmax(scores)]
                        # if len(line) < 15:
                        #     # bad line, too short
                        #     i -= 1
                        #     continue
                        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
                        if count(line, string.punctuation) > 6:
                            # bad line, too much punctuation
                            continue

                        last_word = self.strip_punc(line.split()[-1])


                        try:
                            if not last_word.isalpha(): last_word = line.split()[-2]
                        except IndexError:
                            continue

                        if not last_word.isalpha():
                            # bad line, ends in multiple punctuations
                            continue

                        #for ii, l in enumerate(candidate_lines):
                        #    print(l, scores[ii])

                        print("CHOSEN LINE::: {}".format(line), score)
                        if end_word == "\n":
                            end_word = r"\n"
                        tag = " ({} -> {}) ".format(temp_topic_word, end_word)
                        poem_lines.append(line + tag)
                        prime += (line + '\n')

                        i += 1

                    #poem = prime[len(orig_prime):]
                    poem_lines = self.clean_poem(poem_lines)
                    poem = "\n".join(poem_lines)
                    #poem = self.correct_grammar(poem)
                    print ("\n\nLINES WRITTEN BY CANDLELIGHT\n{}".format(poem))
                    topic_word = input("Topic?")
                    self.args.prime = input("Prime?")

                    #continuous = False


        output_path = self.args.output_path
        if not output_path is None:
            if not os.path.isdir(output_path):
                output_path = os.path.join(self.args.save_dir, output_path)
            with open(output_path, "a") as f:
                f.write('\n\n')
                # for item in text_list:
                #     f.write(item)
                f.write(poem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', '-t', type=str, default='cheese',
                        help='topic word for poem')
    parser.add_argument('--n_lines', '-l', type=int, default=4,
                        help='number of lines to generate')
    parser.add_argument('--n_syllables', '-s', type=int, default=10,
                        help='number of syllables per line')

    # syllables = 8
    # n_lines = 9
    #
    args = parser.parse_args()

    PRIME = "creamy milky kurds and feta cheese,\n"
    PRIME = ""
    args.topic = "majestic mountain"


    pw = PoemWriter(save_dir = save_dir, prime= PRIME)
    pw.sample(args.n_syllables, args.n_lines, args.topic)
