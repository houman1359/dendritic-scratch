"""
networks.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.

The conductance based gradient scaling can be activated by setting
gradient_strategy: block_conductance_dynamic
"""

from copy import deepcopy
from torch import nn
from typing import List
import torch
import torch.nn.functional as F
from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs
from dendritic_modeling import logger


class ExcitationInhibitionLayer(nn.Module):
    def __init__(
        self, 
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors, 
        inhibitory_branch_factors,
        excitatory_input_dim, 
        ee_synapses_per_branch, 
        ei_synapses_per_branch, 
        inhibitory_input_dim = None, 
        ie_synapses_per_branch = None, 
        ii_synapses_per_branch = None, 
        reactivate = True, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting = True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type='param_tanh',
        enable_branch_outputs=False 
    ):
        super(ExcitationInhibitionLayer, self).__init__()

        if enable_branch_outputs:
            self.inhibitory_cells = DendriNetWithOutputs(
                n_soma = n_inhibitory_cells, 
                branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ei_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, 
                inhibitory_synapses_per_branch = ii_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses, 
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
            )
            self.excitatory_cells = DendriNetWithOutputs(
                n_soma = n_excitatory_cells, 
                branch_factors = excitatory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ee_synapses_per_branch,
                inhibitory_input_dim = n_inhibitory_cells, 
                inhibitory_synapses_per_branch = ie_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
            )
        else:
            self.inhibitory_cells = DendriNet(
                n_soma = n_inhibitory_cells, 
                branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ei_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, 
                inhibitory_synapses_per_branch = ii_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses, 
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
            )
            self.excitatory_cells = DendriNet(
                n_soma = n_excitatory_cells, 
                branch_factors = excitatory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ee_synapses_per_branch,
                inhibitory_input_dim = n_inhibitory_cells, 
                inhibitory_synapses_per_branch = ie_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
            )
    
    def decay_weights(self, weight_decay):
        self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if not hasattr(self.inhibitory_cells, 'forward_with_branch_outputs'):
            inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
            excitatory_output = self.excitatory_cells(x, inhibitory_output)
            return excitatory_output, inhibitory_output, [], []
        else:
            inhibitory_final, inh_acts = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
            excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(x, inhibitory_final)
            return excitatory_final, inhibitory_final, inh_acts, exc_acts


class ExcitationInhibitionNetwork(nn.Module):
    def __init__(
        self, 
        input_dim, 
        excitatory_layer_sizes, 
        inhibitory_layer_sizes,
        excitatory_branch_factors, 
        inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, 
        ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, 
        ii_synapses_per_branch_per_layer = [], 
        reactivate = False, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting=True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        weight_decay_rate=0.1,
        reactivation_type='param_tanh',
        local_loss_weight=0.0,
        enable_branch_outputs=False, 
        output_layer=False,
        output_dim=10,
        learning_strategy="mle",
    ):
        super(ExcitationInhibitionNetwork, self).__init__()
        self.weight_decay_rate = weight_decay_rate
        self.n_layers = len(excitatory_layer_sizes)
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.local_loss_weight = local_loss_weight
        self.learning_strategy = learning_strategy

        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)

        ee_syn = list(ee_synapses_per_branch_per_layer)
        ei_syn = list(ei_synapses_per_branch_per_layer)
        ie_syn = list(ie_synapses_per_branch_per_layer)
        ii_syn = list(ii_synapses_per_branch_per_layer)
        while len(ee_syn) < self.n_layers:
            ee_syn.append(ee_syn[-1])
        while len(ei_syn) < self.n_layers:
            ei_syn.append(ei_syn[-1])
        while len(ie_syn) < self.n_layers:
            ie_syn.append(ie_syn[-1])
        while len(ii_syn) < self.n_layers:
            ii_syn.append(None)

        layers = []
        for i in range(self.n_layers):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = excitatory_layer_sizes[i+1], 
                n_inhibitory_cells = inhibitory_layer_sizes[i+1],
                excitatory_branch_factors = excitatory_branch_factors, 
                inhibitory_branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_layer_sizes[i], 
                ee_synapses_per_branch = ee_syn[i], 
                ei_synapses_per_branch = ei_syn[i],
                inhibitory_input_dim = inhibitory_layer_sizes[i],
                ie_synapses_per_branch = ie_syn[i], 
                ii_synapses_per_branch = ii_syn[i],
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_strategy = self.reactivation_strategy,
                blocklinear_strategy = self.blocklinear_strategy,
                reactivation_type = reactivation_type,
                enable_branch_outputs = enable_branch_outputs
            ))
        
        self.layers = nn.ModuleList(layers)

        self.feedback_mats = nn.ParameterList()
        for i in range(self.n_layers):
            if self.learning_strategy == "local_credit_assignment":
                fb = nn.Parameter(torch.randn(excitatory_layer_sizes[i+1], 10)*0.01, requires_grad=False)
                self.feedback_mats.append(fb)
            else:
                self.feedback_mats.append(None)

    def decay_weights(self, weight_decay=None):
        """
        Called by the optimizer step to apply weight decay to each layer.
        If weight_decay is not provided, we default to self.weight_decay_rate.
        """
        if weight_decay is None:
            weight_decay = self.weight_decay_rate
        for layer in self.layers:
            layer.decay_weights(weight_decay)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for i, layer in enumerate(self.layers):
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory

    def forward_with_branch_outputs(self, x):
        excitatory = x
        inhibitory = None
        all_acts = []
        for i, layer in enumerate(self.layers):
            excitatory, inhibitory, inh_acts, exc_acts = layer.forward_with_branch_outputs(excitatory, inhibitory)
            all_acts.extend(exc_acts)
        return excitatory, all_acts

    def compute_local_loss(self, x, y=None):
        return 0.0

    @property
    def branch_layers(self):
        """
        Provide a combined list of DendriticBranchLayer modules from all
        excitatory sub-nets so that code expecting `model.branch_layers`
        won't crash. Each item is a DendriticBranchLayer, typically
        found in excitatory_cells.branch_layers.
        """
        all_branch_layers = []
        for ei_layer in self.layers:
            all_branch_layers.extend(ei_layer.excitatory_cells.branch_layers)
        return all_branch_layers

    @property
    def n_branch_layers(self):
        """
        Provide the number of branch layers. For code referencing
        `model.n_branch_layers`, we match DendriNet's definition by counting
        how many DendriticBranchLayer modules the excitatory sub-nets hold.
        """
        return len(self.branch_layers)

############################################################
# MLPExcInhNetwork
############################################################
class MLPExcInhLayer(nn.Module):
    def __init__(self, in_excit, in_inhib, out_features, activation=None):
        super(MLPExcInhLayer, self).__init__()
        self.in_excit = int(in_excit)  # → new: ensure int
        self.in_inhib = int(in_inhib)
        self.out_features = int(out_features)
        if self.in_excit > 0:
            self.excit_pre_w = nn.Parameter(torch.zeros(self.out_features, self.in_excit))
            nn.init.xavier_normal_(self.excit_pre_w)
        else:
            self.excit_pre_w = None
        if self.in_inhib > 0:
            self.inhib_pre_w = nn.Parameter(torch.zeros(self.out_features, self.in_inhib))
            nn.init.xavier_normal_(self.inhib_pre_w)
        else:
            self.inhib_pre_w = None
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.activation = activation  # Optional activation function

    def decay_weights(self, weight_decay):
        with torch.no_grad():
            if self.excit_pre_w is not None:
                self.excit_pre_w.sub_(weight_decay * self.excit_pre_w)
            if self.inhib_pre_w is not None:
                self.inhib_pre_w.sub_(weight_decay * self.inhib_pre_w)
            self.bias.sub_(weight_decay * self.bias)

    def forward(self, x):
        # Assume x is of shape [batch, in_excit + in_inhib]
        batch_size = x.size(0)
        if self.in_excit > 0 and self.in_inhib > 0:
            x_excit = x[:, :self.in_excit]
            x_inhib = x[:, self.in_excit:self.in_excit + self.in_inhib]
        elif self.in_excit > 0:
            x_excit = x
            x_inhib = None
        else:
            x_excit = None
            x_inhib = x
        out = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        if self.excit_pre_w is not None and x_excit is not None:
            # Force excitatory weights to be positive
            w_excit = self.excit_pre_w.exp()
            out = out + x_excit.matmul(w_excit.t())
        if self.inhib_pre_w is not None and x_inhib is not None:
            # Force inhibitory weights to be negative
            w_inhib = -self.inhib_pre_w.exp()
            out = out + x_inhib.matmul(w_inhib.t())
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

# → New: MLPExcInhNetwork: builds an MLP mimicking the excitatory/inhibitory structure.
class MLPExcInhNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        excitatory_layer_sizes,         # list of neuron counts for excitatory network (excluding input_dim)
        inhibitory_layer_sizes,           # list of inhibitory neuron counts (if any; can be empty)
        excitatory_branch_factors,        # list of branch factors for excitatory network
        inhibitory_branch_factors,        # list of branch factors for inhibitory network (can be empty)
        ee_synapses_per_branch_per_layer, # list of excitatory synapses per branch for each layer
        ie_synapses_per_branch_per_layer, # list of inhibitory synapses per branch for each layer
        output_layer=False,               # if True, add a final classifier layer
        output_dim=10,
        activation=nn.ReLU(),             # default activation for hidden layers
        weight_decay_rate=0.1,
        learning_strategy="mle",
        **kwargs
    ):
        super(MLPExcInhNetwork, self).__init__()
        logger.info("Building MLPExcInhNetwork...")
        self.weight_decay_rate = weight_decay_rate
        self.learning_strategy = learning_strategy
        self.output_layer_flag = output_layer

        # → Cast configuration values explicitly to int
        input_dim = int(input_dim)
        excitatory_layer_sizes = [int(x) for x in excitatory_layer_sizes]
        inhibitory_layer_sizes = [int(x) for x in inhibitory_layer_sizes] if inhibitory_layer_sizes else []
        ee_synapses_per_branch_per_layer = [int(x) for x in ee_synapses_per_branch_per_layer]
        ie_synapses_per_branch_per_layer = [int(x) for x in ie_synapses_per_branch_per_layer]

        # → Compute effective excitatory input dimensions per layer.
        excit_sizes = deepcopy(excitatory_layer_sizes)
        excit_sizes.insert(0, input_dim)
        self.n_layers = len(excit_sizes) - 1  # number of MLP layers

        layers = []
        for i in range(1, len(excit_sizes)):
            # Compute excitatory inputs: previous layer neurons * excitatory synapses per branch.
            excit_count = excit_sizes[i-1] * ee_synapses_per_branch_per_layer[i-1]
            # Compute inhibitory inputs if provided.
            if inhibitory_layer_sizes:
                inhib_count = inhibitory_layer_sizes[i-1] * ie_synapses_per_branch_per_layer[i-1]
            else:
                inhib_count = 0
            in_dim = excit_count + inhib_count
            layer_activation = activation if i < self.n_layers else None  # no activation on final layer
            layer_mod = MLPExcInhLayer(in_excit=excit_count, in_inhib=inhib_count, out_features=excit_sizes[i], activation=layer_activation)
            layers.append(layer_mod)
        self.layers = nn.ModuleList(layers)

        # → If output_layer is true, add a final classifier layer.
        if self.output_layer_flag:
            self.classifier = nn.Linear(int(excit_sizes[-1]), int(output_dim))
        else:
            self.classifier = None

    def decay_weights(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.weight_decay_rate
        for layer in self.layers:
            layer.decay_weights(weight_decay)
        if self.classifier is not None:
            with torch.no_grad():
                self.classifier.weight.sub_(weight_decay * self.classifier.weight)
                self.classifier.bias.sub_(weight_decay * self.classifier.bias)

    def forward(self, x):
        # For the first layer, assume x is excitatory only. Pad inhibitory zeros if needed.
        for i, layer in enumerate(self.layers):
            in_excit = layer.in_excit
            in_inhib = layer.in_inhib
            if x.size(1) < in_excit + in_inhib:
                pad_size = (in_excit + in_inhib) - x.size(1)
                x = torch.cat([x, torch.zeros(x.size(0), pad_size, device=x.device)], dim=1)
            x = layer(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

