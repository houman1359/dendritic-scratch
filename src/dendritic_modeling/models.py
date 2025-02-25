"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from dendritic_modeling.networks import ExcitationInhibitionNetwork

class BaseModel(nn.Module):
    def __init__(self, net):
        super(BaseModel, self).__init__()
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")
    
    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")


class ProbabilisticClassifier(BaseModel):
    """
    A probabilistic classifier that outputs a categorical distribution
    via LogSoftmax. 
    """
    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # if we want a final FC, create it here
        if self.use_output_layer:
            self.final_fc = nn.Linear(net.forward(torch.zeros(1, net.n_branch_layers)).shape[1], self.output_dim)

        self.final = nn.LogSoftmax(dim = -1)
        self.log_output_scale = nn.Parameter(torch.zeros((self.output_dim,)), 
                                             requires_grad = True)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)
    
    def forward(self, x, y=None, sampler=False):
        # net_output is shape [batch_size, last_excit_size]
        net_output = self.net(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)  # optional

        final_logits = self.final(self.log_output_scale.exp() * net_output)
        categorical_dist = Categorical(logits=final_logits)
        if y is None:
            return categorical_dist if sampler else categorical_dist.logits
        else:
            nll = -categorical_dist.log_prob(y)
            return nll
        
    def predict(self, x, stochastic=False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)
        
    def compute_loss(self, x, y):
        nll = self.forward(x, y=y)
        main_loss = nll.mean(dim=0)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts

    def decay_weights(self, weight_decay):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.net, 'decay_weights'):
            self.net.decay_weights(weight_decay)


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits; 
    cross-entropy is used for compute_loss.
    """
    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(Classifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        if self.use_output_layer:
            self.final_fc = nn.Linear(self.output_dim, self.output_dim) 
            
    def forward(self, x, y=None):
        logits = self.net(x)
        if hasattr(self, 'final_fc'):
            logits = self.final_fc(logits)
        return logits

    def predict(self, x, stochastic=False):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts

    def decay_weights(self, weight_decay):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.net, 'decay_weights'):
            self.net.decay_weights(weight_decay)