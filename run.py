"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os,sys,time,argparse,torch,random,math
import numpy as np
import torch
from torch import optim,nn
from reader import Data,Vocabulary
from model.memnn import KVMMModel
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

def train(input_tensors, target_tensors, kbs, model, model_optimizer, criterion, vocab, kb_vocab):
    model_optimizer.zero_grad()
    input_tensors = torch.from_numpy(np.expand_dims(input_tensors,axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs,axis=0))
    target_tensors = torch.from_numpy(np.expand_dims(target_tensors,axis=0))

    # Teacher forcing: Feed the target as the next input
    output = model(input_tensors[0], kbs[0])
    output=output.type(torch.FloatTensor)
    target_tensors = target_tensors[0]
    output = output.permute(0,2,1)
    _,target_maxvals = target_tensors.max(2)
    loss = criterion(output, target_maxvals)
    loss.backward()
    model_optimizer.step()
    return loss.item()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent == 0:
        percent = 0.001
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(model, validation_inputs, validation_targets, kbs):
    with torch.no_grad():
        input_tensors = torch.from_numpy(np.expand_dims(validation_inputs,axis=0))
        target_tensors = torch.from_numpy(np.expand_dims(validation_targets,axis=0))
        kbs = torch.from_numpy(np.expand_dims(kbs,axis=0))
        model.batch_size = input_tensors[0].shape[0]
        output = model(input_tensors[0], kbs[0])
        _,outputmax = output.max(2)
        target_tensors = target_tensors[0]
        outputmaxnp = outputmax.cpu().numpy()
        target_tensorsnp = target_tensors.cpu().numpy()
        accuracy = float(np.sum(outputmaxnp == target_tensorsnp))/(input_tensors[0].shape[0] * input_tensors[0].shape[1])
        model.batch_size = batch_size
        return accuracy

def main(args):
    # Dataset functions
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
                  batch_size=batch_size,
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  encoder_units=200,
                  decoder_units=200).to(device)

    print(model)
    model_optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = 100
    start = time.time() 
    n_iters = 1000000

    iter = 0
    while iter < n_iters:
        training_data = training.generator(batch_size)
        input_tensors = training_data[0][0]
        target_tensors = training_data[1]
        kbs = training_data[0][1]
        iter += 1
        loss = train(input_tensors, target_tensors, kbs, model, model_optimizer, criterion, vocab, kb_vocab)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            validation_data = validation.generator(batch_size)
            validation_inputs = validation_data[0][0]
            validation_kbs = validation_data[0][1]
            validation_targets = validation_data[1]
            print("vi = %s,  vt = %s, vk = %s",validation_inputs.shape,validation_targets.shape,validation_kbs.shape)
            accuracy = evaluate(model, validation_inputs, validation_targets, validation_kbs)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f - accuracy %f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, accuracy))
            torch.save(model.state_dict(), 'model_weights_nkbb.hdf5')


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
