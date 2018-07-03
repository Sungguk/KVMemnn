import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reshape(tensor,batch_size,seq_length,embed_size,pad_length):
    tensor = torch.sum(tensor, dim=2)
    tensor= torch.reshape(tensor,(batch_size,embed_size,seq_length))
    return tensor

def reshape2(tensor,batch_size,pad_length,seq_length):
    tensor = tensor[:batch_size, :pad_length,]
    v = torch.zeros((batch_size, pad_length, 1523))
    tensor = torch.reshape(tensor, (batch_size, pad_length,431))
    tensor = torch.cat((v, tensor), dim=2)
    return tensor

def reshaped(tensor,batch_size,pad_length,seq_length):
    tensor = torch.reshape(tensor, (batch_size,pad_length,seq_length))
    return tensor

class KVMMModel(nn.Module):
    def __init__(self, pad_length=20,batch_size=100,embedding_size=200,n_chars=20,vocab_size=1000,n_labels=20,encoder_units=256,decoder_units=256):
        super(KVMMModel, self).__init__()
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars
        self.vocab_size = vocab_size
        self.n_labels =  n_labels
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

    def forward(self, input_dialogues, input_keyvalues):
        #input1: Dialogues
        input_embed1 = nn.Embedding(self.vocab_size, self.embedding_size, self.pad_length)(input_dialogues)
        dropout = nn.Dropout(0.2)(input_embed1)
        encoder = nn.LSTM(self.encoder_units)(dropout)
        decoder = nn.LSTM(self.decoder_units)(encoder)
        dense1 = nn.Sequential(nn.Linear(200),nn.tanh())(encoder)
        dense2 = nn.Sequential(nn.Linear(200),nn.tanh())(decoder)
        dense3 = nn.Sequential(nn.Linear(200),nn.tanh())(torch.add(dense1, dense2)) #equation 2 (refer to https://arxiv.org/pdf/1705.05414.pdf)
        attention = F.softmax(dense3) #equation 3
        n_hidden = torch.mul(attention, encoder) #equation 4  
        output = nn.Linear(1954)(torch.cat((encoder, n_hidden), dim=0)) #equation 5
    
        # input2: Key value table
        input_embed2 = nn.Embedding(self.vocab_size, self.embedding_size,431)(input_keyvalues)
        input_embed2 = reshape(input_embed2, self.batch_size, 431, self.embedding_size, self.pad_length)
        decoder = reshaped(decoder, self.batch_size, self.pad_length, self.decoder_units)
        n_dense1 = nn.Sequential(nn.Linear(20),nn.tanh())(input_embed2)
        n_dense1 = reshaped(n_dense1, batch_size, self.pad_length,  self.decoder_units)
        n_dense2 = nn.Sequential(nn.Linear(200),nn.tanh())(decoder)
        n_dense3 = nn.Sequential(nn.Linear(431),nn.tanh())(torch.cat(n_dense1, n_dense2, 0)) #equation 7
        n_dense3 = reshape2(n_dense3, self.batch_size,  pad_length, 431)
        n_out = torch.add(output, n_dense3) # equation 8
        n_output = F.softmax(n_out) # equation 9
        return n_output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



