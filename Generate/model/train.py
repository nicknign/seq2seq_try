#!/usr/bin/python
# -*- coding: utf-8 -*
import os
import sys
import tensorflow as tf
from seq2seq_model import Seq2Seqmodel
from util import *

filepath = os.path.split(os.path.realpath(__file__))[0]
datapath = "{}/../data/simple_dialog.txt".format(filepath)

reload(sys)
sys.setdefaultencoding('utf8')


###============= prepare data
encoder_metadata, decoder_metadata, idx_q, idx_a = load_data(datapath)

(trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)

trainX = trans_idx(trainX, encoder_metadata["w2idx"])
trainY = trans_idx(trainY, decoder_metadata["w2idx"])
testX = trans_idx(testX, encoder_metadata["w2idx"])
testY = trans_idx(testY, decoder_metadata["w2idx"])
validX = trans_idx(validX, encoder_metadata["w2idx"])
validY = trans_idx(validY, decoder_metadata["w2idx"])

###============= parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_integer("max_grad_norm", 30, "max_grad_norm.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("embedding_dimension", 128 , "embedding dimension")
tf.flags.DEFINE_integer("seq2seq_layer", 1, "seq2seq_layer.")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("epochs", 20, "Number of epochs to train for.")


xseq_len = len(trainX)
yseq_len = len(trainY)
assert xseq_len == yseq_len

xvocab_size = len(encoder_metadata['idx2w'])
yvocab_size = len(decoder_metadata['idx2w'])

FLAGS = tf.flags.FLAGS

flags = {'learning_rate': FLAGS.learning_rate,
         'max_grad_norm': FLAGS.max_grad_norm,
         'batch_size': FLAGS.batch_size,
         'embedding_dimension': FLAGS.embedding_dimension,
         'seq2seq_layer': FLAGS.seq2seq_layer,
         'dropout_keep_prob': FLAGS.dropout_keep_prob,
         'epochs': FLAGS.epochs}

print("parameters:\n train seq_len:{}\nbatch size:{}\nxvocab size:{}\nyvocab size:{}\nembedding size:{}".format(
xseq_len, flags['batch_size'], xvocab_size, yvocab_size, flags['embedding_dimension']))
print("dropout_keep_prob:{}\nepochs:{}".format(flags['dropout_keep_prob'], flags['epochs']))


###============= model
Model = Seq2Seqmodel("./seq2seq")
Model.saveDict(encoder_metadata, decoder_metadata)
Model.load_train_data(trainX, trainY, testX, testY, validX, validY)
Model.trainModel(flags)

test_sentences = [u"你好啊"]
answer = Model.predictSeq(test_sentences)
print("ask:{}".format(test_sentences))
print("answer:")
for a in answer:
    print(a)
while (1):
    senten = raw_input("ask>>>")
    answer = Model.predictSeq([senten])
    print("answer:")
    for a in answer:
        print(a)