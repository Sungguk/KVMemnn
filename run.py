"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os,sys,time,argparse,torch,random,math
import numpy as np
from torch import optim,nn
from reader import Data,Vocabulary
from model.memnn import KVMMModel
from torchsummary import summary
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

def train(input_tensors, target_tensors, kbs, model, model_optimizer, criterion):
    input_tensors = torch.from_numpy(np.expand_dims(input_tensors,axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs,axis=0))
    target_tensors = torch.from_numpy(np.expand_dims(target_tensors,axis=0))

    # Teacher forcing: Feed the target as the next input
    output = model(input_tensors[0], kbs[0])
    output=output.type(torch.FloatTensor)
    target_tensors = target_tensors[0]
    output = output.permute(0,2,1)
    loss = criterion(output, target_tensors)
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
                  batch_size=100,
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  encoder_units=200,
                  decoder_units=200).to(device)

    print(model)
    model_optimizer = optim.SGD(model.parameters(), lr=0.1)
    model_optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = 10
    start = time.time() 
    n_iters = 1000000

    iter = 0
    while iter < n_iters:
        ind = random.randint(0,len(training.inputs)-1) 
        if (ind+batch_size >= len(training.inputs)):
            continue
        input_tensors = training.inputs[ind:ind+batch_size]
        target_tensors = training.targets[ind:ind+batch_size]
        iter += 1
        loss = train(input_tensors, target_tensors, np.repeat(training.kbs[np.newaxis,:,:],batch_size,axis=0), model, model_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


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
