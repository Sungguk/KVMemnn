"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os,sys
import argparse
import numpy as np
import torch
from torch import optim,nn
from reader import Data,Vocabulary
from model.memnn import KVMMModel
from torchsummary import summary
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

MAX_LENGTH = 20

def train(input_tensor, target_tensor, kbs, model, model_optimizer, criterion, max_length=MAX_LENGTH):
#    model_hidden = model.initHidden()

    model_optimizer.zero_grad()

    input_length = input_tensor.size
    target_length = target_tensor.size


    model_outputs = torch.zeros(max_length, 1, device=device)

    loss = 0
    input_tensor = torch.from_numpy(np.expand_dims(input_tensor,axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs,axis=0))
    output = model(input_tensor, kbs)
#
#    decoder_input = torch.tensor([[SOS_token]], device=device)
#
#    decoder_hidden = encoder_hidden
#
#    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#
#    if use_teacher_forcing:
#        # Teacher forcing: Feed the target as the next input
#        for di in range(target_length):
#            decoder_output, decoder_hidden, decoder_attention = decoder(
#                decoder_input, decoder_hidden, encoder_outputs)
#            loss += criterion(decoder_output, target_tensor[di])
#            decoder_input = target_tensor[di]  # Teacher forcing
#
#    else:
#        # Without teacher forcing: use its own predictions as the next input
#        for di in range(target_length):
#            decoder_output, decoder_hidden, decoder_attention = decoder(
#                decoder_input, decoder_hidden, encoder_outputs)
#            topv, topi = decoder_output.topk(1)
#            decoder_input = topi.squeeze().detach()  # detach from history as input
#
#            loss += criterion(decoder_output, target_tensor[di])
#            if decoder_input.item() == EOS_token:
#                break
#
#    loss.backward()
#
#    encoder_optimizer.step()
#    decoder_optimizer.step()
#    return loss.item() / target_length


def main(args):
    # Dataset functions
    vocab = Vocabulary('./data/vocabulary.json', padding=args.padding)
    vocab = Vocabulary('./data/vocabulary.json',
                              padding=args.padding)
    kb_vocab=Vocabulary('./data/vocabulary.json',
                              padding=4)
    print('Loading datasets.')
    training = Data(args.training_data, vocab,kb_vocab)
    validation = Data(args.validation_data, vocab, kb_vocab)
    training.load()
    validation.load()
    training.transform()
    training.kb_out()
    validation.transform()
    validation.kb_out()
    print('Datasets Loaded.')
    print('Compiling Model.')

    model = KVMMModel(pad_length=args.padding,
                  embedding_size=args.embedding,
                  vocab_size=vocab.size(),
                  batch_size=1,
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  encoder_units=200,
                  decoder_units=200).to(device)

    print(model)
    model_optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()


    for iter in range(1, 2):
        input_tensor = training.inputs[iter-1]
        target_tensor = training.targets[iter-1]
        loss = train(input_tensor, target_tensor, training.kbs, model, model_optimizer, criterion)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=130, type=int)
    named_args.add_argument('-es', '--embedding', metavar='|',
                            help="""Size of the embedding""",
                            required=False, default=200, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/train_data.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/val_data.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=100, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
