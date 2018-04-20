import re
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np
from itertools import groupby
import string
import poetrytools

from beam import BeamSearch

class Model():
    def __init__(self, args, infer=False):

        self.return_dict = {}

        input_dim = 2

        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        BONUS = True if args.bonus else False
        print("BONUS {}".format(BONUS))
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        #self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length, input_dim])
        self.bonus_features = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name = "BonusFeatures")
        self.topic_words = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="Topics")
        self.syllables = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name = "SyllableCount")
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        with tf.variable_scope('rnnlm', reuse=None):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):

                # Create new variable named 'embedding' to connect the character input to the base layer
                # of the RNN. Its role is the conceptual inverse of softmax_w.
                # It contains the trainable weights from the one-hot input vector to the lowest layer of RNN.
                mult = 2 if BONUS else 1
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])

                # Create an embedding tensor with tf.nn.embedding_lookup(embedding, self.input_data).
                # This tensor has dimensions batch_size x seq_length x rnn_size.
                # tf.split splits that embedding lookup tensor into seq_length tensors (along dimension 1).
                # Thus inputs is a list of seq_length different tensors,
                # each of dimension batch_size x 1 x rnn_size.
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1) # substitute embedding with input; split along SEQ in Batch

                if BONUS:
                    bonus_features = tf.split(tf.nn.embedding_lookup(embedding, self.bonus_features), args.seq_length, 1)
                    topic_words = tf.split(tf.nn.embedding_lookup(embedding, self.topic_words), args.seq_length, 1)

                    # Concat these - 10 cells will be given the last word
                    o = []
                    #print(inputs[0].shape)
                    #print(bonus_features[0].shape)
                    last_word_size = int(args.rnn_size/2)
                    #last_word_size = 128

                    # Syllables [sequences in batch, words in sequence, 1,1] => seqlength, batch_size x 1 x 1:
                    syllables = self.syllables[..., None, None]
                    syllables = tf.cast(tf.transpose(syllables, [1,0,2,3]), tf.float32)

                    for n in range(0,len(inputs)):
                        #o.append(tf.concat([inputs[n][:, :, :args.rnn_size*mult-last_word_size], bonus_features[n][:, :, :last_word_size]], 2))
                        if args.use_topics:
                            o.append(tf.concat([inputs[n], bonus_features[n], topic_words[n], syllables[n]], 2))
                        else:
                            o.append(tf.concat([inputs[n], bonus_features[n], syllables[n]], 2))
                        #o = bonus_features
                    #seq length, batch size x 1 x 2*rnn)
                    inputs = o

                # Iterate through these resulting tensors and eliminate that degenerate second dimension of 1,
                # i.e. squeeze each from batch_size x 1 x rnn_size down to batch_size x rnn_size.
                # Thus we now have a list of seq_length tensors, each with dimension batch_size x rnn_size.
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs] # make it into a list, squeeze removes the 1 dimensional SEQ-in-Batch;
            #self.return_dict["INPUT SHAPE"] = inputs.shape


        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def simple_line_eval(self, lines, end_word, sylls):
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        scores = []
        for line in lines:
            punct_pen = -count(line, string.punctuation)

            try:
                last = line.split()[-1]
                if not last.isalpha(): last = line.split()[-2]
                end_pen = 1 if last == end_word else -1
            except:
                end_pen = -20

            actual_syllables = len(''.join([poetrytools.stress(x, "min") for x in line.split()]))
            syll_pen = -np.abs(actual_syllables - sylls)

            score = punct_pen + 5 * end_pen + syll_pen
            scores.append(score)
        return scores



    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=1, width=4, quiet=False, end_word = "end", syllables = 10, return_line_list = False, topic_word = "\n"):

        # Find endword in vocab
        if type(end_word) == type(""):
            #[args.batch_size, args.seq_length]
            # print("End word: " + end_word)
            #end_word = "monarchise"
            idx = vocab[end_word]
            #end_tensor = np.asarray([[idx]])
            end_tensor = np.zeros((1, 1))                    
            end_tensor[0, 0] = vocab.get(end_word,0)
            topic_tensor = np.zeros((1, 1))
            topic_tensor[0, 0] = vocab.get(topic_word,0)


        syl_tensor = np.zeros((1, 1)) + syllables

        def argmaxn(array, n):
            a = np.copy(array)
            out = []
            for i in range(n):
                idx = np.argmax(a)
                out.append(idx);
                a[idx] = 0
            return out

        def weighted_pick(weights):
            if False:
                w = np.copy(weights)
                top_words = []
                wts = []
                for x in argmaxn(w,4):
                    top_words.append(words[x])
                    wts.append(w[x])
                    w[x] = 5 * w[x]
                print("Top words: {}".format(top_words))
                print("Weights: {}".format(wts))
            else:
                w = weights

            t = np.cumsum(w) # you basically make a line, where the width of each word is the probability of that word
            s = np.sum(w)
            prev_word_idx = 0 if len(chosen_words)==0 else vocab[chosen_words[-1]] # don't randomly pick the same word 2x
            chosen = prev_word_idx
            while chosen == prev_word_idx:
                chosen = (int(np.searchsorted(t, np.random.rand(1)*s)))
            return chosen

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            #x = np.zeros((512, 50))
            x[0, 0] = sample[-1]

            # Keep feeding one rhyming word UNTIL newline space is sampled, then feed next
            # Need to adjust beam search to go line by line UGH!

            feed = {self.input_data: x, self.initial_state: state, self.bonus_features: end_tensor, self.syllables: syl_tensor, self.topic_words : topic_tensor}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def create_bs(prime):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys())) # pick a random prime word if needed
            prime_labels = [vocab.get(word, 0) for word in prime.split()] # tokenize prime words
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)), # reset state?
                            prime_labels) # pass labels?
            return bs, prime_labels

        def beam_search_pick(prime, width):
            total_sample = 0
            ret = prime + ' '
            eol = None
            # "\n"
            if False:
                while total_sample < num:
                    bs, prime_labels = create_bs(prime)
                    samples, scores, states = bs.search(None, eol, k=width, maxsample=1000)
                    # Choose
                    sample_choice = np.argmin(scores)
                    chosen = samples[sample_choice]
                    next_words = ""
                    for i, label in enumerate(chosen):
                        next_words += words[label] + ' '
                    ret = next_words
                    total_sample += i - len(prime_labels)
                    prime = ret
            else:
                bs, prime_labels = create_bs(prime)
                samples, scores, states= bs.search(None, None, k=width, maxsample=num)
                pred = samples[np.argmin(scores)]
                for i, label in enumerate(pred):
                    ret += ' ' + words[label] if i > 0 else words[label]
            return ret

        ret = ''
        chosen_ps = []
        chosen_words = []

        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
            if not quiet:
                print(prime)

            # prime network
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word,0)
                feed = {self.input_data: x, self.initial_state:state, self.bonus_features : end_tensor, self.syllables: syl_tensor, self.topic_words : topic_tensor}
                [state] = sess.run([self.final_state], feed)

            ret = ""
            word = prime.split()[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state, self.bonus_features : end_tensor, self.syllables: syl_tensor, self.topic_words : topic_tensor}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]
                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p) # sample from everywhere
                    else:
                        #sample = weighted_pick(p)
                        sample = np.argmax(p) # p is just the list of probabilities
                        #p.argsort()[-10:][::-1]
                        #top_args = self.argmaxn(p, 2)
                        #sample = top_args[np.random.randint(0, len(top_args))] # sample from top 10 words

                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                if sample > len(p): sample -= 1

                chosen_ps.append(p[sample])
                chosen_words.append(words[sample])
                pred = words[sample]
                ret += ' ' + pred
                if pred == '\n':
                    chosen_ps.append("|")
                word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            ret = pred

        if return_line_list and pick != 2: # don't do it on the beam search
            lines = [l for l in ret.split("\n") ]
            score_list = [list(group) for k, group in groupby(chosen_ps, lambda x: x == "|") if not k]
            # print(score_list)

            ### THE Following is the Statistical Evaluation of each Line.
            # output_score = []
            # for i, l in enumerate(lines):
            #     if i < len(score_list):
            #         s = score_list[i]
            #         # ignore most surprising word
            #         # s = [m for m in s if m < .8] # ignore obvious over 8
            #         if len(s) > 10:
            #             s_trim = sorted(s)[2:-3] # ignore least common, and top 3, end word, end punc, new line
            #         else:
            #             s_trim = s
            #         score = np.product(s_trim)**(1./len(s_trim))
            #         if not re.search("[-.,;:]+ ?(and)? ?" + end_word, l) is None:
            #             score -= .1
            #         output_score.append(score)
            #         #print(l)
            #         #print(s)
            #         # print(l + " {:4.2f} ".format(score))
            #     else: # don't score if bad index
            #         pass

            scores = self.simple_line_eval(lines, end_word, syllables)
            # ind = np.random.choice(range(len(lines)-1))
            return lines[np.argmax(scores)], max(scores)


        return ret
