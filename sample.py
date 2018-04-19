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


# Prime - first word
# Update model to prime with end words
def main(save_dir='save', n=200, prime = ' ', count = 1, end_word = "turtle", output_path = "sample.txt", internal_call = False, model = None, syllables = 10, pick = 1, width = 4, sampling_type = 2, return_line_list = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', '-s', type=str, default=save_dir,
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=n,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default=prime,
                       help='prime text')
    parser.add_argument('--pick', type=int, default=pick,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=5,
                       help='width of the beam search')
    parser.add_argument('--sample', type=int, default=sampling_type,
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
    parser.add_argument('--return_line_list', '-r', default=return_line_list,
                       help='0 - Return lines as a list of lines', type=int)

    if internal_call:
        args = parser.parse_args("")
        #sample2(args, model_dict = model)
        sample(args)
    else:
        args = parser.parse_args()
        sample(args)

def sample2(args, model_dict):
    sess = model_dict["sess"]
    model = model_dict["model"]
    words = model_dict["words"]
    vocab = model_dict["vocab"]
    samp = model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width, args.quiet, args.end_word, args.syllables)

    output_path = args.output_path
    if not output_path is None:
        if not os.path.isdir(output_path):
            output_path=os.path.join(args.save_dir, output_path)
    with open(output_path, "a") as f:
        f.write('\n\n')
        f.write(samp)
    
def sample(args):
        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
            if sys.version_info[0] >= 3:
                words, vocab = cPickle.load(f, encoding = 'latin-1')
            else:
                words, vocab = cPickle.load(f)
            
            model = Model(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            text_list = []

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for _ in range(args.count):
                    s = model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width, args.quiet, args.end_word, args.syllables, args.return_line_list)
                    print(s)
                    text_list.append(s)
        output_path = args.output_path
        if not output_path is None and not args.return_line_list:
            if not os.path.isdir(output_path):
                output_path=os.path.join(args.save_dir, output_path)
            with open(output_path, "a") as f:
                f.write('\n\n')
                for item in text_list:
                    f.write(item)
                
if __name__ == '__main__':
        main(save_dir = r"./save/MASTER", end_word="turtle", output_path = "sample.txt", syllables = 10, prime = "lest i paint the gentle sea amidst this tempestuous storm", pick = 1, n = 500, width = 3, sampling_type = 1, return_line_list = True)
        # Sampling type - 2 - weighted sample first word of each line
        #                 1 - weighted sample all
        #                 0 - best word

# Basic rhyming sketch:
    # Sample one line - evaluate
    # [Sample as necessary]
    # Sample next line -- attempt to rhyme with end of first line using (datamuse)
        # Verify word is in vocabulary
        # If failed , repeat

    # metaphor injection:
        # ask for metaphor or generate a pair of words
        # OR ask for a topic
        #  - get relationship from metaphor magnet
        # prime line with words from the magnet, try to end the rhyme with the other word???
        # Substitute word patterns with the metaphor
            # Once a pattern is found, get next metaphor