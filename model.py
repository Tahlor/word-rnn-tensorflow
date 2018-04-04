import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np

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

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=1, width=4, quiet=False, end_word = "end", syllables = 10):

        # Find endword in vocab
        if type(end_word) == type(""):
            #[args.batch_size, args.seq_length]
            print("End word: " + end_word)
            #end_word = "monarchise"
            idx = vocab[end_word]
            #end_tensor = np.asarray([[idx]])
            end_tensor = np.zeros((1, 1))                    
            end_tensor[0, 0] = vocab.get(end_word,0)

        syl_tensor = np.ones((1, 1)) * syllables

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

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

            feed = {self.input_data: x, self.initial_state: state, self.bonus_features: end_tensor, self.syllables: syl_tensor}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        ret = ''
        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
            if not quiet:
                print(prime)
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word,0)
                feed = {self.input_data: x, self.initial_state:state, self.bonus_features : end_tensor}
                [state] = sess.run([self.final_state], feed)

            ret = prime
            word = prime.split()[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state, self.bonus_features : end_tensor}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = words[sample]
                ret += ' ' + pred
                word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            for i, label in enumerate(pred):
                ret += ' ' + words[label] if i > 0 else words[label]
        return ret
