import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Joint_network(nn.Module):
    
    def __init__(self, trans_output_size, pred_output_size, target_size, hidden_size=0, dropout_ratio=0.):
        super(Joint_network, self).__init__()

        self.trans_size = trans_output_size
        self.pred_size = pred_output_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio else None
        
        if hidden_size:
            self.jt_nn = nn.Linear(trans_output_size + pred_output_size, hidden_size)
            project_size = hidden_size
        else:
            self.jt_nn = None
            project_size = trans_output_size + pred_output_size
    
        self.dense = nn.Linear(project_size, target_size)

    def forward(self, trans_output, pred_output):
        '''

        args:
            trans_output: float32 2D Tensor with shape [batch, trans_size], values are the outputs of the trancription network.
            pred_output: float 32 2D Tensor with shape [batch, pred_size], values are the outputs of the prediction network.        

        return:
            joint_outpout: float32 2D Tensor with shape [batch, target_size], values are logits.
        '''   

        input = torch.cat([trans_output, pred_output], dim=1)

        if self.dropout is not None:
            input = self.dropout(input)

        projection = torch.tanh(self.jt_nn(input)) if self.jt_nn is not None else input

        joint_output = self.dense(projection)

        return joint_output

class Separate_network(nn.Module):
    
    def __init__(self, trans_output_size, pred_output_size, target_size, dropout_ratio=0.):
        super(Separate_network, self).__init__()

        self.trans_size = trans_output_size
        self.pred_size = pred_output_size
        self.target_size = target_size
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio else None
        
        self.trans_project_nn = nn.Linear(trans_output_size, target_size)
        self.pred_project_nn = nn.Linear(pred_output_size, target_size)
    
    def forward(self, trans_output, pred_output):
        '''

        args:
            trans_output: float32 2D Tensor with shape [batch, trans_size], values are the outputs of the trancription network.
            pred_output: float 32 2D Tensor with shape [batch, pred_size], values are the outputs of the prediction network.        

        return:
            (trans_project, pred_project): tuple each is a float32 2D Tensor with shape [batch, target_size], values are logits.
        '''   

        if self.dropout is not None:
            trans_input = self.dropout(trans_output)
            pred_input = self.dropout(pred_output)

        trans_project = self.trans_project_nn(trans_input)
        pred_project = self.pred_project_nn(pred_input)

        return (trans_project, pred_project)


class Scorer(nn.Module):
    
    def __init__(self, net, norm_type):
        super(Scorer, self).__init__()

        self.net = net
        self.norm_type = norm_type

    def forward(self, trans_output, pred_output):

        if self.norm_type == 'local-joint':
            jt_output = self.net(trans_output, pred_output)
            activation = nn.LogSoftmax(dim=1)
            return activation(jt_output)

        elif self.norm_type == 'local-separate':
            trans_project, pred_project = self.net(trans_output, pred_output)
            jt_output = (trans_project + pred_project)
            activation = nn.LogSoftmax(dim=1)
            return activation(jt_output)            
        
        elif self.norm_type == 'global-joint-normal':
            jt_output = self.net(trans_output, pred_output)            
            activation = None
            return jt_output

        elif self.norm_type == 'global-joint-logsoftmax':
            jt_output = self.net(trans_output, pred_output)
            activation = nn.LogSoftmax(dim=1)
            return activation(jt_output)

        elif self.norm_type == 'global-separate-normal':
            trans_project, pred_project = self.net(trans_output, pred_output)
            jt_output = (trans_project + pred_project)
            activation = None
            return jt_output

        elif self.norm_type == 'global-separate-logsoftmax':
            trans_project, pred_project = self.net(trans_output, pred_output)
            activation = nn.LogSoftmax(dim=1)
            jt_output = (activation(trans_project) + activation(pred_project))
            return jt_output      

        