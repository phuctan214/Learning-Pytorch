import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy # tokenizer
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger= spacy.load("de_core_news_sm")
spacy_eng = spacy.load("de_core_news_sm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenizr_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

german = Field(tokenize= tokenizr_ger, lower= True,
               init_token="<sos>", eos_token="<eos>")

english = Field(tokenize= tokenize_eng, lower= True,
                init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(exts=('.de_core_news_sm','.de_core_news_sm'),
                                                         fields= (german,english))

german.build_vocab(train_data,max_size = 10000, min_freq = 2)
english.build_vocab(train_data,max_size = 10000, min_freq = 2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout= p)

    def forward(self,x):
        embedding = self.dropout(self.embedding(x))
        
        outputs, (hidden, cell) = self.rnn(embedding)
        
        return hidden,cell
    
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, vocab_size, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self,x, hidden, cell):
        x = torch.unsqueeze(x,0)
        x = self.embedding(x)
        outputs, (hidden,cell) = nn.LSTM(x,(hidden,cell))

        outputs = torch.squeeze(outputs,0)
        pred = self.fc(outputs)

        return pred, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder: Encoder, decoder: Decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, target, source, teacher_force_ratio =0.5):
        batch_size = target.shape[1]
        target_length = source.shape[0]
        vocab_size = len(english.vocab)

        outputs = torch.ones(target_length, batch_size, vocab_size).to(device)
        hidden, cell = self.encoder(source)

        x = target[0]
        for i in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden,cell)
            outputs[i] = output

            best_guess = torch.argmax(output, 1)
            x = target[i] if random.random() < teacher_force_ratio else best_guess


        return outputs

#Hyperparameters
num_epoch = 1
lr_rate = 1e-4
batch_size = 64

#Model Hypeparameters
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
embedding_size_encoder = 300
embedding_size_decoder = 300
hidden_size = 1024
num_layers = 2
p = 0.5

#TensorBroad
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = batch_size,
    sort_within_bacth = True,
    sort_key = lambda x: len(x.src),
    device = device
)

encoder = Encoder(input_size_encoder, hidden_size, embedding_size_encoder, num_layers,p).to(device)
decoder = Decoder(input_size_decoder, hidden_size, embedding_size_decoder, output_size, num_layers,p).to(device)

model = Seq2Seq(encoder, decoder)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr_rate = lr_rate)

for epoch in range(num_epoch):
    print(f'Epoch [{epoch}/{num_epoch}]')
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    # save_checkpoint(checkpoint)
    for batch_idx, batch in enumerate(train_iterator):
        target = batch.src.to(device)
        input_data = batch.trg.to(device)

        output = model(target,input_data)
        output = output[1:].reshape(-1, output.shape[2])
        target = output[1: ].reshape(-1)

        torch.zero_grad()
        loss = criterion(output,target)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm= 1)
        optimizer.step()

        writer.add_scalar("Tranning loss", loss, global_step=step)
        step = step +1




