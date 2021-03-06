from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

import sample
import subprocess

# Defaults
# python train.py --data_dir ./data --rnn_size 256 --num_layers 2 --model lstm --batch_size 50 --seq_length 25 --num_epochs 50

# "D:\PyCharm Projects\word-rnn-tensorflow\data\poems_large.txt"
def main(data_dir=r".\data\original", rnn_size=256, num_layers=2, model= "gru", batch_size = 50, seq_length = 200, num_epochs=10, save_dir = "save", bonus = False, sample = True, use_topics = True):

    parser = argparse.ArgumentParser()
    parser.add_argument('--bonus', type=str2bool    , default=bonus,
                        help='include extra input features')
    parser.add_argument('--sample', type=str2bool    , default=sample,
                        help='sample periodically')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                       help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default=save_dir,
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=rnn_size,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=num_layers,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default=model,
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=seq_length,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=num_epochs,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=.8,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--end_word_training', type=str2bool, default=False,
                       help='train network to spit out last word')
    parser.add_argument('--syllable_training', type=str2bool, default=False,
                        help='train network to count syllables')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--use_topics', '-t', default=str2bool,
                        help='Use topic words', type=str)

    args = parser.parse_args()

    ##global save_dir
    #save_dir=("./save/BONUS" if BONUS else "save")
    default_save = save_dir
    save_dir = args.save_dir

    # Only number when using default save direectory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif default_save == save_dir:
        n = 1
        while os.path.exists(save_dir+str(n)):
            n += 1
        new_save_dir = save_dir+str(n)
        os.mkdir(new_save_dir)
        save_dir = new_save_dir

    args.save_dir = save_dir
    
    train(args)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def train(args):
    tf.reset_default_graph()
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        try:
            # check if all necessary files exist
            assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
            assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
            ckpt = tf.train.get_checkpoint_state(args.init_from)
            assert ckpt,"No checkpoint found"
            assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

            # open old config and check if models are compatible
            with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
                saved_model_args = cPickle.load(f)
            need_be_same=["model","rnn_size","num_layers","seq_length"]
            for checkme in need_be_same:
                assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

            # open saved vocab/dict and check if vocabs/dicts are compatible
            with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
                saved_words, saved_vocab = cPickle.load(f)
            assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
            assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
        except:
            print("Could not init from old file")

    ## Dump new stuff
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model_dict = {"model":model, "words":data_loader.words, "vocab":data_loader.vocab, "sess":sess} 
        train_writer.add_graph(sess.graph)

        # Write graph quick
        writer = tf.summary.FileWriter(os.path.join(args.save_dir, "graph"), sess.graph)
        writer.close()

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.init_from is not None:
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                print("Could not restore")
            
        # Epoch loop    
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                try:
                    data_loader.pointer = model.batch_pointer.eval()
                    args.init_from = None
                except:
                    pass

            # Batch step loop
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y, last_words, syllables, topic_words = data_loader.next_batch()

                # Concatenate Inputs
                #x = tf.concat([x[:,:,None],last_words[:,:,None]],2)
                if args.end_word_training:
                    feed = {model.input_data: x, model.targets: last_words, model.bonus_features: last_words,
                            model.initial_state: state, model.syllables : syllables, model.topic_words : topic_words,
                            model.batch_time: speed}
                elif args.syllable_training:
                    feed = {model.input_data: x, model.targets: last_words, model.bonus_features: last_words,
                            model.initial_state: state, model.syllables : syllables, model.topic_words : topic_words,
                            model.batch_time: speed}
                else:
                    feed = {model.input_data: x, model.targets: y, model.bonus_features: last_words,
                            model.initial_state: state, model.syllables : syllables, model.topic_words : topic_words,
                            model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                #if (e * data_loader.num_batches + b) % args.save_every == 0 \
                #if b % 1000 in [1, 100] \
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                                    
                    #sample.main(save_dir = args.save_dir, output_path = "sample.txt", internal_call = True, model = model_dict)
                    python_path = "python"
                    #python_path = r"/usr/bin/python2.6/python"
                    if args.sample:
                        subprocess.call("python sample.py -e turtle -o sample.txt -s {}".format(args.save_dir).split(), shell=False)
                    
                    
        train_writer.close()

if __name__ == '__main__':
    main(bonus = True, sample = False, data_dir=r"./data/test", batch_size = 50, seq_length = 60, num_epochs = 200, rnn_size=256)
