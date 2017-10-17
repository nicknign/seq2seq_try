#!/usr/bin/python
# -*- coding: utf-8 -*
import jieba as jb
import copy
import re
import random
import tqdm
from sklearn.utils import shuffle

MAX_HIS = 150
UNK_RANDOM_RATE = 30

def cut_string(string):
    lis = []
    wordit = jb.cut(string)
    for word in wordit:
        if word != " ":
            lis.append(word)
    return lis

def insert_unk(question):
    random_int = random.randint(0, UNK_RANDOM_RATE)
    if random_int == 1:
        random_int = random.randint(0, len(question) - 2)
        question[random_int] = u"UNK"
    return question

def load_data(filepath):
    encoder_metadata = {'idx2w':[u"PAD", u"UNK", u"EOS"], 'w2idx':{}}
    decoder_metadata = {'idx2w':[u"PAD", u"SOS", u"EOS"], 'w2idx':{}}

    with open(filepath, "r") as fp:
        lines = fp.readlines()
        answer_flag = False
        answer = ""
        inputstr = ""
        cmd = ""
        old_gid = None
        Q = []
        A = []
        question = []
        respond = []
        with tqdm.tqdm(total=len(lines)) as pbar:
            for line in lines:
                pbar.update(1)

                if line.startswith("gid:"):
                    gid = line[4:].strip()
                elif line.startswith("input:"):
                    inputstr = line[6:].strip()
                elif line.startswith("cmd:"):
                    cmd = line[4:].strip()
                elif line.startswith("answer:"):
                    answer = line[7:].strip()
                    answer_flag = True
                elif line.startswith("engine:"):
                    if cmd.find("<engine> <transmisson> <autowarranty>") != -1:
                        question = []
                        respond = []
                        continue

                    if old_gid is None:
                        old_gid = gid
                    if gid != old_gid:
                        question = []
                        old_gid = gid

                    # question = []

                    question.extend(cut_string(inputstr))
                    question.append(u"EOS")
                    question = insert_unk(question)
                    cmd_li = re.findall(u"\".+?\"", unicode(cmd))
                    if cmd.find("COMBINE") != -1:
                        cmd_li = [u"COMBINE", u"EOS"]
                    if cmd_li:
                        temp = [i.strip("\"").decode('unicode-escape') for i in cmd_li]
                        respond.extend(temp)
                        respond.append(u"EOS")
                    else:
                        respond.extend(cut_string(answer))
                        respond.append(u"EOS")

                    for w in question:
                        if w not in encoder_metadata['idx2w']:
                            encoder_metadata['idx2w'].append(w)
                    for w in respond:
                        if w not in decoder_metadata['idx2w']:
                            decoder_metadata['idx2w'].append(w)

                    Q.append(copy.deepcopy(question))
                    A.append(respond)
                    question.extend(respond)
                    question = question[-MAX_HIS:]
                    respond = []
                    answer_flag = False
                elif answer_flag:
                    answer += line.strip()
                    continue

    for index, word in enumerate(encoder_metadata['idx2w']):
        encoder_metadata['w2idx'][word] = index

    for index, word in enumerate(decoder_metadata['idx2w']):
        decoder_metadata['w2idx'][word] = index

    return encoder_metadata, decoder_metadata, Q, A

def split_dataset(x, y):
    ratio = [0.8, 0.1, 0.1]
    # number of examples
    data_len = len(x)
    lens = [int(data_len * item) for item in ratio]
    x, y = shuffle(x, y, random_state=0)
    trainX, trainY = x[:lens[0] + lens[1]], y[:lens[0] + lens[1]]
    validX, validY = x[lens[0]:lens[0] + lens[1]], y[lens[0]:lens[0] + lens[1]]
    testX, testY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)

def trans_idx(seqs, w2idx):
    idx_seqs = []
    for seq in seqs:
        idx_seq = []
        for word in seq:
            idx = w2idx[word]
            idx_seq.append(idx)
        idx_seqs.append(idx_seq)
    return idx_seqs

























