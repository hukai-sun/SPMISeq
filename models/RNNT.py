import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import Transcription_networks, Prediction_network
from .scorer import Scorer, Joint_network, Separate_network
from .searcher import Greedy_Searcher, Beam_Searcher
from hparams import hparams

class RNNT(nn.Module):
    def __init__(self, hparams, num_words, num_chars, num_labels, embedd_word_table, embedd_char_table, embedd_label_table, device, norm_type):
        super(RNNT, self).__init__()

        hp = hparams
        
        self.TransNet = Transcription_networks(hp.word_dim, num_words, hp.char_dim, num_chars, hp.num_filters, hp.kernel_size, \
                        hp.Trans_hidden_size, hp.Trans_rnn_layers, embedd_word_table, embedd_char_table, hp.dropout_ratio) 
    
        self.PredNet = Prediction_network(num_labels, hp.Pred_hidden_size, \
                        hp.Pred_rnn_layers, embedd_label_table)

        if norm_type == 'local-joint':
            self.JTNet = Joint_network(hp.Trans_hidden_size * 2, hp.Pred_hidden_size, num_labels, hp.jt_hidden_size, hp.dropout_ratio)
        elif norm_type == 'local-separate':
            self.JTNet = Separate_network(hp.Trans_hidden_size * 2, hp.Pred_hidden_size, num_labels, hp.dropout_ratio)
            
        self.ScoreNet = Scorer(self.JTNet, norm_type=norm_type)
        self.nll_loss = nn.NLLLoss(reduction='none')

        self.device = device
        self.sos = hp.sos
        self.num_labels = num_labels 

    def trans_forward(self, input_word, input_char, mask=None, length=None, hx=None):
        return self.TransNet.forward(input_word, input_char, mask, length, hx)        
    
    def pred_forward(self, input_label, length=None, hx=None):
        return self.PredNet.forward(input_label, length, hx)

    def score_forward(self, trans_output, pred_output):
        return self.ScoreNet(trans_output, pred_output)

    def loss(self, input_word, input_char, input_label, target, mask, length, trans_hx=None, pred_hx=None):

        trans_output, _, mask, length = self.trans_forward(input_word, input_char, mask, length, trans_hx)
        pred_output = self.pred_forward(input_label, length, pred_hx)

        max_len = length.max()

        trans_hidden_size = trans_output.size(-1)
        pred_hidden_size = pred_output.size(-1)

        projection = self.score_forward(trans_output.contiguous().view(-1, trans_hidden_size), pred_output.contiguous().view(-1, pred_hidden_size))

        target = target[:, :max_len].contiguous()

        return (self.nll_loss(projection, target.view(-1)) * mask.contiguous().view(-1)).sum() / mask.size(0)


    def greedy_search(self, input_word, input_char, mask, length, trans_hx=None, pred_hx=None):
        
        with torch.no_grad():

            trans_output, _, _, length = self.trans_forward(input_word, input_char, mask, length, trans_hx)
            
            Searcher = Greedy_Searcher(self.PredNet, self.ScoreNet, self.device, self.sos)
            fw_results = Searcher.search_forward(trans_output, length)
            bw_results = Searcher.search_backward(fw_results)

            return bw_results

    def beam_search(self, beam_width, input_word, input_char, mask, length, trans_hx=None, pred_hx=None):
        
        with torch.no_grad():

            trans_output, _, _, length = self.trans_forward(input_word, input_char, mask, length, trans_hx)

            Searcher = Beam_Searcher(beam_width, self.PredNet, self.ScoreNet, self.device, self.num_labels, self.sos)
            fw_results, fw_bps, fw_scores = Searcher.search_forward(trans_output, length)
            bw_results = Searcher.search_backward(fw_results, fw_bps, fw_scores, length)

            return bw_results









