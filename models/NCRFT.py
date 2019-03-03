import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import Transcription_networks, Prediction_network
from .scorer import Scorer, Joint_network, Separate_network
from .searcher import Greedy_Searcher, Beam_Searcher

class NCRFT(nn.Module):
    def __init__(self, hparams, num_words, num_chars, num_labels, embedd_word_table, embedd_char_table, embedd_label_table, device, norm_type):
        super(NCRFT, self).__init__()

        hp = hparams
        
        self.TransNet = Transcription_networks(hp.word_dim, num_words, hp.char_dim, num_chars, hp.num_filters, hp.kernel_size, \
                        hp.Trans_hidden_size, hp.Trans_rnn_layers, embedd_word_table, embedd_char_table, hp.dropout_ratio) 
    
        self.PredNet = Prediction_network(num_labels, hp.Pred_hidden_size, \
                        hp.Pred_rnn_layers, embedd_label_table)

        if norm_type == 'global-joint-normal' or norm_type == 'global-joint-logsoftmax':
            self.JTNet = Joint_network(hp.Trans_hidden_size * 2, hp.Pred_hidden_size, num_labels, hp.jt_hidden_size, hp.dropout_ratio)
        elif norm_type == 'global-separate-normal' or norm_type == 'global-separate-logsoftmax':
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

    def loss(self, input_word, input_char, input_label, sample_trace, target, sample_target, mask, length, trans_hx=None, pred_hx=None):
        '''

        args:
            input_word: int64 Tensor with shape [batch, lens].
            input_char: int64 Tensor with shape [batch, lens, max_char_len].
            input_label: int64 Tensor with shape [batch, lens].
            sample_trace: int64 Tensor with shape [batch, num_samples, seq_lens], where seq_lens is the maximum of the length in this minibatch. 
            target: int64 Tensor with shape [batch, lens].
            sample_target: int64 Tensor with shape [batch, num_samples, seq_lens].
            mask: float32 Tensor with shape [batch, lens], 0. for paddings.
            length: int64 Tensor with shape [batch], true length of each sample in the minibatch.
            trans_hx: initialization of the trans rnn.
            pred_hx: initialization of the pred rnn.

        return:
            loss: float32 Tensor.
            sample_loss: float32 Tensor.
            data_loss: float32 Tensor.
        '''

        sample_trace_size = sample_trace.size()
        batch_size = sample_trace_size[0]
        num_samples = sample_trace_size[1]
        max_len = sample_trace_size[2]


        data_trans_output, _, data_mask, _ = self.trans_forward(input_word, input_char, mask, length, trans_hx)
        data_pred_output = self.pred_forward(input_label, length, pred_hx)

        trans_hidden_size = data_trans_output.size(-1)
        pred_hidden_size = data_pred_output.size(-1)

        sample_trace = sample_trace.contiguous().view(-1, max_len)
        sample_mask = data_mask.repeat(1, num_samples).contiguous().view(batch_size * num_samples, -1)

        sample_pred_output = self.pred_forward(sample_trace, None, None)

        data_projection = self.score_forward(data_trans_output.contiguous().view(-1, trans_hidden_size), \
            data_pred_output.contiguous().view(-1, pred_hidden_size))

        rep_sample_trans = data_trans_output.contiguous().view(batch_size, -1).repeat(1, num_samples).contiguous().view(-1, trans_hidden_size)
        sample_projection = self.score_forward(rep_sample_trans, sample_pred_output.contiguous().view(-1, pred_hidden_size))

        target = target[:, :max_len].contiguous().view(-1)
        data_loss = -self.nll_loss(data_projection, target) * data_mask.contiguous().view(-1) # [batch * max_len]
        data_loss = data_loss.sum() / batch_size


        sample_target = sample_target.contiguous().view(-1)
        sample_loss = -self.nll_loss(sample_projection, sample_target) * sample_mask.contiguous().view(-1) #[batch * num_samples * max_len]
        sample_loss = sample_loss.contiguous().view(batch_size, num_samples, max_len)

        sample_loss = torch.logsumexp(torch.sum(sample_loss, dim=2), dim=1)
        sample_loss = torch.mean(sample_loss, dim=0)

        loss = sample_loss - data_loss

        return loss, data_loss , sample_loss

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