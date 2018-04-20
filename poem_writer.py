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
import metaphor as meta

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
TOP_VOCAB_WORDS = 20000
NUM_OF_SAMPLES = 10

class PoemWriter():

    def __init__(self, save_dir='save', n=50, prime = ' ', count = 1, end_word = "turtle", output_path = "sample.txt", internal_call = False, model = None, syllables = 10, pick = 1, use_topics = False):

        if pick == 1:
            self.number_of_samples = NUM_OF_SAMPLES
        else:
            self.number_of_samples = 1

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

    def write_out_poem(self, poem, path=None):
        output_path = self.args.output_path
        if not output_path is None:
            if not os.path.isdir(output_path):
                output_path = os.path.join(self.args.save_dir, output_path)
            with open(output_path, "a") as f:
                f.write('\n\n')
                # for item in text_list:
                #     f.write(item)
                f.write(poem)

    def sample(self, num_syllables, num_lines, topic_word, custom_rhyme = [], related_words=[], metaphor = ""):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
            text_list = []

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                continuous = True
                while continuous:
                    related_words_muse = datamuser.get_all_related_words(topic_word.split(), TOP_TOPIC_WORDS)
                    # Get related words
                    if metaphor != "":
                        related_words = meta.main(metaphor)
                    elif related_words == []:
                        related_words = related_words_muse[:]

                    if type(related_words) == type([]):
                        related_words = set(related_words)
                    print(related_words)
                    topic_words = list(related_words.intersection(self.freq_words))
                    if len(topic_words) == 0:
                        raise ValueError("No vocab words related to topic")
                    print (topic_words)

                    # Do priming
                    prime = ""
                    if self.args.prime == "":
                        if metaphor != "":
                            prime = metaphor
                        else:
                            for ii in range(0, int(num_syllables/2)):
                                prime += random.choice(topic_words) + " "
                    else:
                        prime = self.args.prime

                    if prime[-1] != "\n":
                        prime += ".\n"

                    """print("Pre-prime: {}".format(prime))
                    # make sure prime words are valid
                    p = prime.lower()
                    p = p.replace("\n", "|")
                    p = [x for x in p.split() if x in self.words]
                    prime = " ".join(p).replace("|", "\n")"""

                    # prime = '{} {} {}\n'.format(topic_word, topic_word, topic_word)
                    quiet = True
                    poem_lines = []
                    print("Prime {}".format(prime))
                    i = 0
                    while i < num_lines:
                        # get endword

                        if i % 2 == 0 and custom_rhyme == []:
                            # pick random topic word
                            end_word = random.choice(topic_words)
                            #end_word = "\n"
                        else:
                            # rhyme with last line
                            if custom_rhyme != []:
                                last_word = custom_rhyme[i % len(custom_rhyme)]

                            # don't try to rhyme if doing a metaphor
                            if metaphor == "":
                                rhymes = list(datamuser.get_rhymes(last_word, weak_rhymes=False).intersection(self.freq_words))
                            else:
                                rhymes = list(related_words_muse.intersection(self.freq_words))

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
                        for j in range(self.number_of_samples):  # get best of 10 lines
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
                    self.write_out_poem(poem)

                    if False:
                        topic_word = input("Topic?")
                        new_prime = input("Prime?")
                        related_words = [] # reset this
                        if new_prime == "same":
                            pass
                        else:
                            self.args.prime = new_prime

                    #continuous = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', '-t', type=str, default='cheese',
                        help='topic word for poem')
    parser.add_argument('--n_lines', '-l', type=int, default=4,
                        help='number of lines to generate')
    parser.add_argument('--n_syllables', '-s', type=int, default=8  ,
                        help='number of syllables per line')

    # syllables = 8
    # n_lines = 9
    #
    args = parser.parse_args()

    PRIME = "creamy milky kurds and feta cheese,\n"
    PRIME = ""
    args.topic = "road"

    PRIME = """Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;

Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,

And both that morning equally lay
In leaves no step had trodden black.
Oh, I kept the first for another day!
Yet knowing how way leads on to way,
I doubted if I should ever come back.

I shall be telling this with a sigh
Somewhere ages and ages hence:
Two roads diverged in a wood, and I"""
    RHYME = ["sigh", "hence"]
    RHYME = []
    args.topic = "cheese"
    PRIME = args.topic
    RELATED_WORDS = ["creamey", "cheesy", "milk", "gooey", "cheddar", "feta", "culture", "cheese", "mold", "food", "delicious", "flavour"]
    RELATED_WORDS = ["cheese", "food", "delicious", "flavour"]

    METAPHOR = r"marriage as death"
    METAPHOR = r"student as beggar"
    args.topic = "scholar"
    PRIME=""
    pw = PoemWriter(save_dir = save_dir, prime= PRIME, pick=1)
    pw.sample(args.n_syllables, args.n_lines, args.topic, custom_rhyme=RHYME, related_words=RELATED_WORDS, metaphor=METAPHOR)
