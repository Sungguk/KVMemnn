import torch
import torch.nn as nn
import torch.nn.functional as F

#Addiing support for CUDA tensor types (utilize GPUs for computation) if not available use  CPU tensors.

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

#Key-Value Memory Network
class KVMMModel(nn.Module):
    def __init__(self, pad_length=20,batch_size=100,embedding_size=200,n_chars=20,vocab_size=1000,n_labels=20,encoder_units=256,decoder_units=256):

        #Initialize variables
		super(KVMMModel, self).__init__()

        self.pad_length = pad_length
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars
        self.vocab_size = vocab_size
        self.n_labels =  n_labels
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
      
        #Initialize the models
        #For dialogue input
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
        dropout = self.dialogue_dropout(input_embed1)
        encoder = self.encoder_dialogue(dropout)
        decoder = self.decoder_dialogue(encoder[0])

        #All equations are refered to https://arxiv.org/pdf/1705.05414.pdf
        #Implementation of equation 2
        dense1 = self.dense1_dialogue(encoder[0])#apply tanh on after applying linear transformation on encoder output 
        dense2 = self.dense2_dialogue(decoder[0])#apply tanh on after applying linear transformation on decoder output 
        dense3 = self.dense3_dialogue(torch.add(dense1, dense2))#combine the output from dens1 and dense2, then applying linear trnsformation and tanh funtion.  
		
        attention = F.softmax(dense3, dim=2) #Apply attention function of the dense3. #Implementation of equation3
        n_hidden = torch.mul(attention, encoder[0]) #Weighted summerize the result with attention output and sencode output. #Implementation of equation 4  
        output = self.dialogue_output(torch.cat((encoder[0], n_hidden),dim=2)) #Linear transformation on the output of cascating encoder output and n_hidden #Implementation of equation 5
    
        # input2: Key value table
        if device == torch.device("cuda"):
            input_embed2 = self.input_embed_keyvalue(input_keyvalues.cuda())
        else:
            input_embed2 = self.input_embed_keyvalue(input_keyvalues)

		#Implementation of equation 7
		input_embed2 = reshape(input_embed2, self.batch_size, 431, self.embedding_size, self.pad_length)
        n_dense1 = self.keyvalue_dense1(input_embed2)#apply tanh on after applying linear transformation on input_embed2 
        n_dense1 = reshaped(n_dense1, self.batch_size,  self.pad_length, self.decoder_units)
        decoder = reshaped(decoder[0], self.batch_size, self.pad_length, self.decoder_units)
        n_dense2 = self.keyvalue_dense2(decoder)#apply tanh on after applying linear transformation on decoder
        n_dense3 = self.keyvalue_dense3(torch.cat((n_dense1, n_dense2), dim=1))#apply tanh on after applying linear transformation on cascating result of n_dense1 n_dense1 and n_dense2 
        
	#Implementaiton of equation 8
	n_dense3 = reshape2(n_dense3, self.batch_size,  self.pad_length, 431)
        n_out = torch.add(output, n_dense3)#Summerize the n_dense3 with the output from the dialogue part. 
        
        return n_out

  
