#!/usr/bin/python
# -*- coding: utf-8 -*
import tensorflow as tf
import tensorlayer as tl
import pickle
import json
import os
import sys
from tensorlayer.layers import *
from util import cut_string
from fuzzywuzzy import fuzz

reload(sys)
sys.setdefaultencoding('utf8')

MAX_SLEN = 30
FUZZ_TH = 70

class Seq2Seqmodel(object):
    def __init__(self, path):
        self.paras_path = path
        self.encoder_metadata = {}
        self.decoder_metadata = {}
        # self.sess = tf.InteractiveSession()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    def buildModel(self, flags):
        ###============= model
        def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
            with tf.variable_scope("model", reuse=reuse):
                # for chatbot, you can use the same embedding layer,
                # for translation, you may want to use 2 seperated embedding layers
                with tf.variable_scope("encode_embedding") as vs:
                    net_encode = EmbeddingInputlayer(
                        inputs=encode_seqs,
                        vocabulary_size=self.xvocab_size,
                        embedding_size=flags['embedding_dimension'],
                        name='en_embedding')
                    vs.reuse_variables()
                    tl.layers.set_name_reuse(True)
                with tf.variable_scope("decode_embedding") as vs:
                    net_decode = EmbeddingInputlayer(
                        inputs=decode_seqs,
                        vocabulary_size=self.yvocab_size,
                        embedding_size=flags['embedding_dimension'],
                        name='de_embedding')
                    vs.reuse_variables()
                    tl.layers.set_name_reuse(True)
                net_rnn = Seq2Seq(net_encode, net_decode,
                                  cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                  n_hidden=flags['embedding_dimension'],
                                  initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                  encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                                  decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
                                  initial_state_encode=None,
                                  dropout=(flags['dropout_keep_prob'] if is_train else None),
                                  n_layer=flags['seq2seq_layer'],
                                  return_seq_2d=True,
                                  name='seq2seq')
                net_out = DenseLayer(net_rnn, n_units=self.yvocab_size, act=tf.identity, name='output')
            return net_out, net_rnn

        # model for training
        self.encode_seqs = tf.placeholder(dtype=tf.int64, shape=[flags['batch_size'], None], name="encode_seqs")
        self.decode_seqs = tf.placeholder(dtype=tf.int64, shape=[flags['batch_size'], None], name="decode_seqs")
        self.target_seqs = tf.placeholder(dtype=tf.int64, shape=[flags['batch_size'], None], name="target_seqs")
        self.target_mask = tf.placeholder(dtype=tf.int64, shape=[flags['batch_size'], None],
                                     name="target_mask")  # tl.prepro.sequences_get_mask()
        net_out, _ = model(self.encode_seqs, self.decode_seqs, is_train=True, reuse=False)

        # model for inferencing
        self.encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
        self.decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
        self.net, self.net_rnn = model(self.encode_seqs2, self.decode_seqs2, is_train=False, reuse=True)
        self.y = tf.nn.softmax(self.net.outputs)

        # loss for training
        print(net_out.outputs)    # (?, 8004)
        print(self.target_seqs)    # (32, ?)
        # loss_weights = tf.ones_like(target_seqs, dtype=tf.float32)
        # loss = tf.contrib.legacy_seq2seq.sequence_loss(net_out.outputs, target_seqs, loss_weights, yvocab_size)
        self.loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=self.target_seqs,
                                                   input_mask=self.target_mask, return_details=False, name='cost')
        net_out.print_params(False)

        lr = flags['learning_rate']
        # self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        # Truncated Backpropagation for training (option)
        max_grad_norm = flags['max_grad_norm']
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, net_out.all_params),max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        self.train_op = optimizer.apply_gradients(zip(grads, net_out.all_params))
        tl.layers.initialize_global_variables(self.sess)

    def trainModel(self, flags):
        self.saveflags(flags)
        self.buildModel(flags)
        ###============= train
        print("------model train start------")
        start_id = 1
        end_id = 2
        n_epoch = flags['epochs']
        n_step = int(len(self.trainX) / flags['batch_size'])
        for epoch in range(n_epoch):
            ## shuffle training data
            from sklearn.utils import shuffle
            trainX, trainY = shuffle(self.trainX, self.trainY, random_state=0)
            ## train an epoch
            total_err, n_iter = 0, 1
            epoch_time = time.time()
            step_time = time.time()
            for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=flags['batch_size'], shuffle=False):
                X = tl.prepro.pad_sequences(X)
                _target_seqs = tl.prepro.pad_sequences(Y)
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

                _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=True)
                _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)

                _, err = self.sess.run([self.train_op, self.loss],
                                  {self.encode_seqs: X,
                                   self.decode_seqs: _decode_seqs,
                                   self.target_seqs: _target_seqs,
                                   self.target_mask: _target_mask})

                if n_iter % flags['print_epochs'] == 0:
                    print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (
                    epoch + 1, n_epoch, n_iter, n_step, err, time.time() - step_time))
                    step_time = time.time()

                total_err += err
                n_iter += 1
            print("---Epoch[%d/%d] averaged loss:%f took:%.5fs--"
                  % (epoch + 1, n_epoch, total_err / n_iter, time.time() - epoch_time))
            print("Valid:")
            self.validTest(self.validX, self.validY)
            print("Test:")
            self.validTest(self.testX, self.testY)
            print("---------------------------------------------------------")

            tl.files.save_npz(self.net.all_params, name='{}/net.npz'.format(self.paras_path), sess=self.sess)
        print("------model train end------")

    def loadModel(self):
        if os.path.exists("{}/encoder_metadata.pkl".format(self.paras_path)) and \
                os.path.exists("{}/decoder_metadata.pkl".format(self.paras_path)):
            with open('{}/encoder_metadata.pkl'.format(self.paras_path), 'r') as fp:
                self.encoder_metadata = pickle.load(fp)
                self.xvocab_size = len(self.encoder_metadata['idx2w'])
            with open('{}/decoder_metadata.pkl'.format(self.paras_path), 'r') as fp:
                self.decoder_metadata = pickle.load(fp)
                self.yvocab_size = len(self.decoder_metadata['idx2w'])
        if os.path.exists("{}/paras.json".format(self.paras_path)) and os.path.exists("{}/net.npz".format(self.paras_path)):
            flags = self.loadflags()
            self.buildModel(flags)
            tl.files.load_and_assign_npz(sess=self.sess, name='{}/net.npz', network=self.net)


    def validTest(self, X, Y):
        # go through the test set step by step, it will take a while.
        start_time = time.time()
        right = 0
        start_id = 0
        end_id = 2
        # reset all states at the begining
        for index, x in enumerate(X):
                # 1. encode, get state
                state = self.sess.run(self.net_rnn.final_state_encode,
                                 {self.encode_seqs2: [x]})
                # 2. decode, feed start_id, get first word
                #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
                o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],
                                    {self.net_rnn.initial_state_decode: state,
                                     self.decode_seqs2: [[start_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=3)
                # 3. decode, feed state iteratively
                y_ = [w_id]
                if w_id != end_id:
                    for _ in range(MAX_SLEN):  # max sentence length
                        o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],
                                            {self.net_rnn.initial_state_decode: state,
                                             self.decode_seqs2: [[w_id]]})
                        w_id = tl.nlp.sample_top(o[0], top_k=2)
                        if w_id == end_id:
                            break
                        y_ = y_ + [w_id]
                if fuzz.ratio(unicode(Y[index]), unicode(y_)) > FUZZ_TH:
                    right += 1

        test_acc = float(right)/len(Y)
        print("Accuracy: %.3f took %.2fs" % (test_acc, time.time() - start_time))


    def predictSeq(self, seeds, asize = 5):
        start_id = 1
        end_id = 2
        responds = []
        for seed in seeds:
            seed_id = [self.encoder_metadata['w2idx'][w] if self.encoder_metadata['w2idx'].get(w) \
                       else self.encoder_metadata['w2idx'][u'UNK'] for w in cut_string(seed)]
            for _ in range(asize):  # 1 Query --> 5 Reply
                # 1. encode, get state
                state = self.sess.run(self.net_rnn.final_state_encode,
                                 {self.encode_seqs2: [seed_id]})
                # 2. decode, feed start_id, get first word
                #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
                o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],
                                    {self.net_rnn.initial_state_decode: state,
                                     self.decode_seqs2: [[start_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=3)
                w = self.decoder_metadata['idx2w'][w_id]
                # 3. decode, feed state iteratively
                sentence = [w]
                if w_id != end_id:
                    for _ in range(MAX_SLEN):  # max sentence length
                        o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],
                                            {self.net_rnn.initial_state_decode: state,
                                             self.decode_seqs2: [[w_id]]})
                        w_id = tl.nlp.sample_top(o[0], top_k=2)
                        w = self.decoder_metadata['idx2w'][w_id]
                        if w_id == end_id:
                            break
                        sentence = sentence + [w]
                respond = u" ".join(sentence)
                responds.append(respond)
        return responds

    def load_train_data(self, trainX, trainY, testX, testY, validX, validY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.validX = validX
        self.validY = validY

    def saveDict(self, encoder_metadata, decoder_metadata):
        self.encoder_metadata = encoder_metadata
        self.xvocab_size = len(encoder_metadata['idx2w'])
        self.decoder_metadata = decoder_metadata
        self.yvocab_size = len(decoder_metadata['idx2w'])
        with open('{}/encoder_metadata.pkl'.format(self.paras_path), 'wb') as f:
            pickle.dump(self.encoder_metadata, f)
        with open('{}/decoder_metadata.pkl'.format(self.paras_path), 'wb') as f:
            pickle.dump(self.decoder_metadata, f)

    def saveflags(self, flags):
        with open("{}/paras.json".format(self.paras_path), "w") as fp:
            json.dump(flags, fp)

    def loadflags(self):
        with open("{}/paras.json".format(self.paras_path), "r") as fp:
            flags = json.load(fp)
        return flags

if __name__ == "__main__":
    Model = Seq2Seqmodel("./seq2seq")
    Model.loadModel()
    test_sentences = [u"你好"]
    answer = Model.predictSeq(test_sentences)
    print("ask:{}".format(test_sentences))
    print("answer:")
    for a in answer:
        print(a)
    while(1):
        senten = raw_input("ask>>>")
        answer = Model.predictSeq([senten])
        print("answer:")
        for a in answer:
            print(a)