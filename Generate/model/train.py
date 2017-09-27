#!/usr/bin/python
# -*- coding: utf-8 -*
import os
import sys
from util import *

filepath = os.path.split(os.path.realpath(__file__))[0]
datapath = "{}/../data/dialog.txt".format(filepath)

reload(sys)
sys.setdefaultencoding('utf8')


encoder_metadata, decoder_metadata, idx_q, idx_a = load_data(datapath)

(trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)

pass