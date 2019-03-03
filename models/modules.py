import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .rnn_pack_helper import prepare_rnn_seq, recover_rnn_seq

class Transcription_networks(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, \
        hidden_size, rnn_layers, embedd_word_table=None, embedd_char_table=None, dropout_ratio=0.):
        super(Transcription_networks, self).__init__()

        self.word_dim = word_dim
        self.num_words = num_words
        self.char_dim = char_dim
        self.num_chars = num_chars
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.embedd_word_table = embedd_word_table
        self.embedd_char_table = embedd_char_table
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio else None

        if embedd_word_table is None:
            self.word_embedd = nn.Embedding(num_words, word_dim)
        else:
            self.word_embedd = nn.Embedding.from_pretrained(embedd_word_table, freeze=False)
    
        if embedd_char_table is None:
            self.char_embedd = nn.Embedding(num_chars, char_dim)
        else:
            self.char_embedd = nn.Embedding.from_pretrained(embedd_char_table, freeze=False)


        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size //2 )


        self.rnn = nn.LSTM(word_dim + num_filters, hidden_size, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout_ratio)
        

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        '''
        args:
            input_word: int64 Tensor with shape [batch, length], values are word ids in the word alphabet.
            input_char: int64 Tensor with shape [batch, length, char_length], values are char ids in the char alphabet.
            mask: float32 Tensor with shape [batch, length], 0. for padding.
            length: int64 Tensor with shape [batch], true lengths.
            hx: initialization of the hidden state.

        return:
            output: float32 Tensor with shape [batch, seq_len, hidden_size*num_directions], where seq_len is the maximum true length in this minibatch.
            hn: tuple (h_n, c_n) float32 Tensor with shape [num_layers*num_directions, batch, hidden_size], values are the hidden state for t=seq_len.
            mask: float32 Tensor with shape [batch, seq_len], 0. for padding, where seq_len is the maximum true length in this minibatch.
            length: int64 Tensor with shape [batch], true lengths.     
        '''
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()
        # [batch, length, word_dim]
        word = self.word_embedd(input_word)
        # [batch, length, char_length, char_dim]
        char = self.char_embedd(input_char)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # apply dropout word on input
        if self.dropout:
            word = self.dropout(word)
            char = self.dropout(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char], dim=2)
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size]
            output, hn = self.rnn(input, hx=hx)

        return output, hn, mask, length


class Prediction_network(nn.Module):
    def __init__(self, num_labels, hidden_size, rnn_layers, embedd_label_table, dropout_ratio=0.):
        super(Prediction_network, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.embedd_label_table = embedd_label_table
        self.dropout_ratio = dropout_ratio

        self.label_embedd = nn.Embedding.from_pretrained(embedd_label_table, freeze=True)

        in_size = num_labels

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=rnn_layers, batch_first=True, bidirectional=False, dropout=dropout_ratio)

    def forward(self, input, length=None, hx=None):
        '''
        args:
            input_label: int64 Tensor with shape [batch, length], values are the label ids in the label alphabet.
            length: int64 Tensor with shape [batch], true length.
            hx: tuple (h_n, c_n) float32 Tensor with shape [num_layers, batch, hidden_size].

        return:
            output: float32 Tensor with shape [batch, seq_len, hidden_size].   
        '''        
        if length is not None: 
            max_len = length.max()
        else:
            max_len = input.size(1)
         
        # [batch, length, label_dim]
        input_label = input[:,:max_len]

        label = self.label_embedd(input_label)
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, _ = prepare_rnn_seq(label, length, hx=hx, masks=None, batch_first=True)
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch, length, hidden_size]
            output, hn = self.rnn(label, hx=hx)


        return output

    def search_step(self, input, hx):
        '''  
        args:
            input: int64 Tensor with shape [batch], values are the label ids in the label alphabet.
            hx: tuple (h_0, c_0) float32 Tensor with shape [num_layers, batch, hidden_size].

        return:
            output: float32 Tensor with shape [batch, hidden_size].
            hn: tuple (h_1, c_1) float32 Tensor with shape [num_layers, batch, hidden_size].       
        '''
        

        label = self.label_embedd(input)
        label = label.unsqueeze(dim=1)

        output, hn = self.rnn(label, hx=hx)
        output = output.squeeze(dim=1)

        return output, hn
        

class Prediction_network_cell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio=0.):
        super(Prediction_network_cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio else None

        self.rnn_cell = nn.LSTMCell(input_size, hidden_size)
    
    def forward(self, input, hx):
        '''
        args:
            input: int64 Tensor with shape [batch, embedding_dim], values are the label embedding vector.
            hx: tuple (h_0, c_0) float32 Tensor with shape [batch, hidden_size].
        
        return:
            output: float32 Tensor with shape [batch, hidden_size].
        '''

        if self.dropout is not None:
            input = self.dropout(input)

        h_1, c_1 = self.rnn_cell(input, hx)

        return (h_1, c_1)
