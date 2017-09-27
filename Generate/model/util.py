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
        question[random_int] = "UNK"
    return question

def load_data(filepath):
    encoder_metadata = {'idx2w':["EOS", "UNK"], 'w2idx':{}}
    decoder_metadata = {'idx2w':["EOS"], 'w2idx':{}}

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
                    if old_gid is None:
                        old_gid = gid
                    if gid != old_gid:
                        question = []
                        old_gid = gid
                    question.extend(cut_string(inputstr))
                    question.append("EOS")
                    question = insert_unk(question)
                    cmd_li = re.findall(u"\".+?\"", unicode(cmd))
                    if cmd_li:
                        temp = [i.strip("\"").decode('unicode-escape') for i in cmd_li]
                        respond.extend(temp)
                        respond.append("EOS")
                    else:
                        respond.extend(cut_string(answer))
                        respond.append("EOS")

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
    ratio = [0.7, 0.15, 0.15]
    # number of examples
    data_len = len(x)
    lens = [int(data_len * item) for item in ratio]
    x, y = shuffle(x, y, random_state=0)
    trainX, trainY = x[:lens[0] + lens[1]], y[:lens[0] + lens[1]]
    validX, validY = x[lens[0]:lens[0] + lens[1]], y[lens[0]:lens[0] + lens[1]]
    testX, testY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)
