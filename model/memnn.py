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
    if device == torch.device("cuda"):
        v = torch.zeros((batch_size, pad_length, 1523)).cuda()
    else:
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
      
        self.input_embed_dialogues = nn.Embedding(self.vocab_size, self.embedding_size, self.pad_length)
        self.dialogue_dropout = nn.Dropout(0.2)
        self.encoder_dialogue = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        self.decoder_dialogue = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        self.dense1_dialogue = nn.Sequential(nn.Linear(200,200),nn.Tanh())
        self.dense2_dialogue = nn.Sequential(nn.Linear(200,200),nn.Tanh())
        self.dense3_dialogue = nn.Sequential(nn.Linear(200,200),nn.Tanh())
        self.dialogue_output = nn.Linear(400,1954)

        self.input_embed_keyvalue = nn.Embedding(self.vocab_size, self.embedding_size,431)
        self.keyvalue_dense1 = nn.Sequential(nn.Linear(431,20),nn.Tanh())
        self.keyvalue_dense2 = nn.Sequential(nn.Linear(200,200),nn.Tanh())
        self.keyvalue_dense3 = nn.Sequential(nn.Linear(200,431),nn.Tanh())

    def forward(self, input_dialogue, input_keyvalues):
        #input1: Dialogues
        if device == torch.device("cuda"):
            input_embed1 = self.input_embed_dialogues(input_dialogue.cuda())
        else:
            input_embed1 = self.input_embed_dialogues(input_dialogue)
        #print "input_embed1",input_embed1.shape
        dropout = self.dialogue_dropout(input_embed1)
        #print "dropout",dropout.shape
        encoder = self.encoder_dialogue(dropout)
        #print "encoder",encoder[0].shape
        decoder = self.decoder_dialogue(encoder[0])
        #print "decoder",decoder[0].shape
        dense1 = self.dense1_dialogue(encoder[0])
        #print "dense1",dense1.shape
        dense2 = self.dense2_dialogue(decoder[0])
        #print "dense2",dense2.shape
        dense3 = self.dense3_dialogue(torch.add(dense1, dense2)) #equation 2 (refer to https://arxiv.org/pdf/1705.05414.pdf)
        #print "dense3",dense3.shape
        attention = F.softmax(dense3, dim=2) #equation 3
        n_hidden = torch.mul(attention, encoder[0]) #equation 4  
        output = self.dialogue_output(torch.cat((encoder[0], n_hidden),dim=2)) #equation 5
    
        # input2: Key value table
        if device == torch.device("cuda"):
            input_embed2 = self.input_embed_keyvalue(input_keyvalues.cuda())
        else:
            input_embed2 = self.input_embed_keyvalue(input_keyvalues)
        input_embed2 = reshape(input_embed2, self.batch_size, 431, self.embedding_size, self.pad_length)
        #print "input_embed2",input_embed2.shape
        n_dense1 = self.keyvalue_dense1(input_embed2)
        #print "n_dense1",n_dense1.shape
        n_dense1 = reshaped(n_dense1, self.batch_size,  self.pad_length, self.decoder_units)
        #print "n_dense1",n_dense1.shape
        decoder = reshaped(decoder[0], self.batch_size, self.pad_length, self.decoder_units)
        #print "decoder",decoder.shape
        n_dense2 = self.keyvalue_dense2(decoder)
        #print "n_dense2",n_dense2.shape
        #print "concat",torch.cat((n_dense1, n_dense2), dim=1).shape
        n_dense3 = self.keyvalue_dense3(torch.cat((n_dense1, n_dense2), dim=1)) #equation 7
        #print "n_dense3",n_dense3.shape
        n_dense3 = reshape2(n_dense3, self.batch_size,  self.pad_length, 431)
        #print "n_dense3",n_dense3.shape
        n_out = torch.add(output, n_dense3) # equation 8
        #print "n_out",n_out.shape
        #n_output = F.softmax(n_out, dim=2) # equation 9
        #print "n_output",n_output.shape
        return n_out

    #def initHidden(self):
    #    return torch.zeros(1, 1, self.hidden_size, device=device)
