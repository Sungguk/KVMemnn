from reader import Data, Vocabulary
import pandas as pd
import os, sys
import argparse
import torch
import numpy as np
from reader import Data,Vocabulary
from model.memnn import KVMMModel
from math import log
from numpy import array
from numpy import argmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                #print(row[j])
                if row[j]<=0:
                    row[j]=0.000000000000000000001
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences
def run_example(model, kbs,vocabulary, text, groundtruth):
    encoded = vocabulary.string_to_int(text)
    input_tensors = torch.from_numpy(np.expand_dims(encoded,axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs,axis=0))
    prediction = model(input_tensors, kbs[0])
    print(prediction[0].shape)
    result=beam_search_decoder(prediction[0].detach().cpu().numpy(),5)
    data=[]
    for seq in result:
        data.append(' '.join(vocabulary.int_to_string(np.array(seq[0]))))
    print('input:',text)
    print('groundtruth:',groundtruth)
    unpaddata = []
    for sentence in data:
        unpaddata.append(sentence.replace('<pad> ',''))
    print('prediction:', ' '.join(vocabulary.int_to_string(prediction[0].max(1)[1].detach().cpu().numpy())))
    return data


def run_examples(model, kbs, vocabulary, examples, groundtruths):
    predicted = []
    input = []
    for example,groundtruth in zip(examples,groundtruths):
        print('~~~~~')
        input.append(example)
        predicted.append(run_example(model, kbs, vocabulary, example, groundtruth))
    return predicted


if __name__ == "__main__":
    pad_length = 20
    df = pd.read_csv("data/test_data.csv")
    inputs = list(df["inputs"])
    outputs = list(df["outputs"])
    vocab = Vocabulary('data/vocabulary.json', padding=pad_length)

    kb_vocabulary = Vocabulary('data/vocabulary.json',padding = 4)

    model = KVMMModel(pad_length=20,
                  embedding_size=200,
                  batch_size=1,
                  vocab_size=vocab.size(),
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  encoder_units=200,
                  decoder_units=200).to(device)
    weights_file = "model_weights.pytorch"
    model.load_state_dict(torch.load(weights_file))

    kbfile = "data/normalised_kbtuples.csv"
    df = pd.read_csv(kbfile)
    kbs = list(df["subject"] + " " + df["relation"])
    # print(kbs[:3])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    kbs = np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data = run_examples(model, kbs,vocab, inputs, outputs)
    df=pd.DataFrame(columns=["inputs","outputs","prediction"])
    d = {'outputs':[],'inputs':[],'u1': [],'u2':[],'u3':[],'u4':[],'u5':[]}
    for i, o, p in zip(inputs, outputs, data):
        d["outputs"].append(str(o))
        d["inputs"].append(str(i))
        for i,preds in enumerate(p):
            d["u"+str(i+1)].append(str(preds))
    df = pd.DataFrame(d)
    df.to_csv("output_kb.csv")
    # print(outputs)

