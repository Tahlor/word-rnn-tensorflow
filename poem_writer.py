from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
import sys
import datamuser
import random


class PoemWriter():

    def __init__(self, save_dir='save', n=50, prime = ' ', count = 1, end_word = "turtle", output_path = "sample.txt", internal_call = False, model = None, syllables = 10, pick = 1):

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

        self.args = parser.parse_args("")

        with open(os.path.join(self.args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
            saved_args.use_topics=args.use_topics

        with open(os.path.join(self.args.save_dir, 'words_vocab.pkl'), 'rb') as f:
            if sys.version_info[0] >= 3:
                self.words, self.vocab = cPickle.load(f, encoding='latin-1')
            else:
                self.words, self.vocab = cPickle.load(f)

            self.model = Model(saved_args, True)


    def evaluate_line(self, line):
        return True  # for now

    def sample(self, num_syllables, num_lines, topic_word):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
            text_list = []

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                prime = '{}\n'.format(topic_word)
                # prime = '{} {} {}\n'.format(topic_word, topic_word, topic_word)
                orig_prime = prime
                quiet = False
                poem_lines = []

                related_words = datamuser.get_all_related_words(topic_word.split())
                topic_words = list(related_words.intersection(set(self.vocab.keys())))
                if len(topic_words) == 0:
                    raise ValueError("No vocab words related to topic")

                print (topic_words)

                i = 0
                while i < num_lines:


                    # get endword

                    if i % 2 == 0:
                        # pick random topic word
                        end_word = random.choice(topic_words)


                    else:
                        # rhyme with last line
                        rhymes = list(datamuser.get_rhymes(last_word, weak_rhymes=True).intersection(set(self.vocab.keys())))
                        # print (len(rhymes))
                        if len(rhymes)==0: end_word = random.choice(topic_words)
                        else: end_word = random.choice(rhymes)
                        print('LAST WORD: {}  END WORD: {}'.format(last_word, end_word))

                    print ('END WORD: {}'.format(end_word))
                    # end_word = 'flag'

                    lines = self.model.sample(sess, self.words, self.vocab, self.args.n,
                                          prime, self.args.sample, self.args.pick,
                                          self.args.width, quiet, end_word, num_syllables)
                    quiet = True
                    line = lines[len(prime):].split('\n')[1]  # strip off prime and keep next single line
                    last_word = line.split()[-1]
                    if not last_word.isalpha(): last_word = line.split()[-2]
                    if not last_word.isalpha(): last_word = line.split()[-3]

                    # evaluate line here
                    keep = self.evaluate_line(line)
                    if not keep:
                        # try a new line, don't change prime
                        i -= 1

                    else:

                        # print(line)
                        # text_list.append(line)
                        poem_lines.append(line)


                        prime += (line + '\n')


                    i += 1

                poem = prime[len(orig_prime):]
                print (poem)


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
    parser.add_argument('--topic', '-t', type=str, default='love',
                        help='topic word for poem')
    parser.add_argument('--n_lines', '-l', type=int, default=8,
                        help='number of lines to generate')
    parser.add_argument('--n_syllables', '-s', type=int, default=8,
                        help='number of syllables per line')

    # syllables = 8
    # n_lines = 9
    #

    args = parser.parse_args()

    pw = PoemWriter()
    pw.sample(args.n_syllables, args.n_lines, args.topic)
