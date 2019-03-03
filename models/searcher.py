import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Greedy_Searcher(object):

    def __init__(self, rnn_class, score_class, device, sos=0):
        super(Greedy_Searcher, self).__init__()

        self.sos = sos # start symbol of the sentence 
        self.device = device
        self.rnn = rnn_class
        self.rnn_layers = self.rnn.rnn_layers
        self.hidden_size = self.rnn.hidden_size
        self.score_class = score_class

    def label_init(self, dtype, size):

        init_label = torch.zeros(size, dtype=dtype, device=self.device)
        init_label.fill_(self.sos)

        return init_label.long()

    def rnn_class_init(self, dtype, size):

        h_0 = torch.zeros(size, dtype=dtype, device=self.device)
        hx = (h_0, h_0)  

        return hx 

    def decision_step(self, score):

        _, topi = score.topk(1)
        input_label = topi.squeeze(dim=1)

        return input_label

    def search_forward(self, trans_output, length):
        """
        args:
            trans_output: float32 Tensor with shape [batch, seq_len, trans_out_size].
            length: int64 Tensor with shape [batch].

        return:
            search_results: int64 Tensor with shape [batch, max_len].
        """

            
        max_len = length.max()
        batch_size = trans_output.size(0)
        
        search_results = trans_output.new_zeros((batch_size, max_len)).long()

        input_label = self.label_init(torch.int64, size=(batch_size))
        hx = self.rnn_class_init(torch.float32, (self.rnn_layers, batch_size, self.hidden_size))
        
        for step in range(max_len):
            pred_output, hx = self.rnn.search_step(input_label, hx)
            score = self.score_class(trans_output[:, step, :], pred_output)

            input_label = self.decision_step(score)

            search_results[:, step] = input_label               

        return search_results

    def search_backward(self, results):

        return results

class Beam_Searcher(object):

    def __init__(self, beam_width, rnn_class, score_class, device, num_labels, sos=0):
        super(Beam_Searcher, self).__init__()

        self.beam_width = beam_width
        self.rnn = rnn_class
        self.rnn_layers = self.rnn.rnn_layers
        self.hidden_size = self.rnn.hidden_size
        self.score_class = score_class
        self.device = device
        self.num_labels = num_labels
        self.sos = sos 

    def label_init(self, dtype, size):

        init_label = torch.zeros(size, dtype=dtype, device=self.device)
        init_label.fill_(self.sos)

        return init_label.long()

    def rnn_class_init(self, dtype, size):

        h_0 = torch.zeros(size, dtype=dtype, device=self.device)
        hx = (h_0, h_0)  

        return hx 

    def decision_step(self, batch_size, score, hx):

        topv, topi = torch.topk(score, self.beam_width)
        class_id = torch.fmod(topi, self.num_labels).long()
        beam_id = torch.div(topi, self.num_labels).long()
        
        input_label = class_id.contiguous().view(-1)
        (h_n,c_n) = hx
        
        batch_offset = (torch.arange(batch_size) * self.beam_width).view(batch_size, 1).to(self.device).long()

        state_id = batch_offset + beam_id
        state_id = state_id.view(-1)
        state_h = torch.index_select(h_n, 1, state_id).view(-1, batch_size * self.beam_width, self.hidden_size)
        state_c = torch.index_select(c_n, 1, state_id).view(-1, batch_size * self.beam_width, self.hidden_size)
        hx = (state_h,state_c)  

        return input_label, hx, topv, class_id, beam_id   

    def search_forward(self, trans_output, length):
        """
        args:
            trans_output: float32 Tensor with shape [batch, seq_len, trans_out_size].
            length: int64 Tensor with shape [batch].
        """

            
        max_len = length.max()
        batch_size = trans_output.size(0)
        
        input_label = self.label_init(torch.int64, (batch_size * self.beam_width))
        hx = self.rnn_class_init(torch.float32, (self.rnn_layers, batch_size * self.beam_width, self.hidden_size))

        search_results = trans_output.new_zeros((batch_size, self.beam_width, max_len)).long()
        search_bps = trans_output.new_zeros((batch_size, self.beam_width, max_len)).long()
        # ignore the inflated copies to avoid duplicate entries in the top k        
        search_scores = torch.zeros((batch_size, self.beam_width, max_len+1), device=self.device)
        search_scores.fill_(-float('Inf'))
        search_scores.index_fill_(1, torch.zeros((1), device=self.device).long(), 0.0)

        for step in range(max_len):
            
            pred_output, hx = self.rnn.search_step(input_label, hx)
            
            rep_trans = trans_output[:, step, :].repeat(1, self.beam_width).contiguous().view(batch_size * self.beam_width, -1)
            score = self.score_class(rep_trans, pred_output)

            score = score.contiguous().view(batch_size, self.beam_width, -1)
            grow_score = search_scores[:, :, step].contiguous().view(batch_size, self.beam_width, 1) + score
            grow_score = grow_score.contiguous().view(batch_size, -1)
            
            input_label, hx, kp_score, left_node, kp_beam = self.decision_step(batch_size, grow_score, hx)

            search_results[:, :, step] = left_node
            search_bps[:, :, step] = kp_beam
            search_scores[:, :, step+1] = kp_score 

        return search_results, search_bps, search_scores

    def search_backward(self, left_nodes, kp_beams, kp_scores, length):

        """
        args:
            left_nodes: int64 Tensor with shape [batch, num_samples, seq_len], where seq_len is the maximum of the true length in this minibatch.
            kp_beams: int64 Tensor with shape [batch, num_samples, seq_len].
            kp_scores: float32 Tensor with shape [batch, num_samples, seq_len+1].
            length: int64 Tensor with shape [batch].

        return :
            search_results: int64 Tensor with shape [batch, num_samples, seq_len].
        """
            
        search_size = left_nodes.size()
        batch_size = search_size[0]
        num_samples = search_size[1]
        seq_lens = search_size[2]

        search_results = left_nodes.new_zeros((batch_size, num_samples, seq_lens)).long()
            
        for n in range(batch_size):

            t = (length[n]-1)
            last_index = torch.arange(num_samples).to(self.device).long()

            search_results[n,:,t] = left_nodes[n,:,t]

            ancestor_index = last_index

            for j in range(length[n]-2, -1, -1):

                ancestor_index = torch.index_select(kp_beams[n,:,j+1], 0, ancestor_index)
                search_results[n,:,j] = torch.index_select(left_nodes[n,:,j], 0, ancestor_index)

        return search_results

