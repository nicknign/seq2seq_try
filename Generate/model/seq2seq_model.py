#!/usr/bin/python
# -*- coding: utf-8 -*
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Seq2Seq(object):
    def __init__(self,
                 input_length=5,
                 input_dim=3,
                 output_dim=4,
                 hidden_dim=24,
                 output_length=3,
                 depth=2,
                 batch_size = 10,
                 dropout = 0.5):
        self.input_length = input_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.depth = depth
        self.dropout = dropout
        self.batch_size = batch_size
        self.model = None

    def buildModel(self):


    def trainModel(self):


    def loadModel(self):


    def predictSeq(self, sentence):