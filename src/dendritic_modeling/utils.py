"""
utils.py
========
This module contains utility functions for data loading, data splitting, 
and model evaluation.
"""

import os
import json
from typing import Optional
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader


class Shaper(object):
    """
    A utility class for reshaping input data to a specified shape.
    """
    def __init__(self, shape):
        self.shape = shape
    
    def reshape(self, x):
        return x.view(*self.shape)

def convert_to_serializable(obj):
    """
    Recursively convert OmegaConf objects to standard Python data types.
    """
    if isinstance(obj, DictConfig):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_dict(dict_obj: dict, save_path: str, fname: str):
    dict_obj = convert_to_serializable(dict_obj)
    full_path = os.path.join(save_path, fname)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    with open(full_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)


def accuracy_score(classifier: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)
    yhat = classifier.predict(inputs)
    accuracy = torch.mean((yhat == labels).float()).item()
    return accuracy


def evaluate_accuracy(
    classifier: torch.nn.Module, 
    train_ds: Optional[torch.utils.data.Dataset] = None, 
    valid_ds: Optional[torch.utils.data.Dataset] = None, 
    test_ds: Optional[torch.utils.data.Dataset] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = 'accuracy',
):
    train_acc = accuracy_score(classifier, *train_ds[:]) if train_ds else None
    valid_acc = accuracy_score(classifier, *valid_ds[:]) if valid_ds else None
    test_acc  = accuracy_score(classifier, *test_ds[:]) if test_ds else None

    if save_path is not None:
        save_dict(
            {
                'train accuracy': train_acc,
                'valid accuracy': valid_acc,
                'test accuracy': test_acc
            },
            save_path,
            filename
        )
        
    return train_acc, valid_acc, test_acc


################################################################################
# 1) RECORD BRANCH ACTIVATIONS
################################################################################
def record_branch_activations(model, dataset, batch_size=64, device='cpu'):
    """
    For a given DendriNetWithOutputs, forward the entire dataset and record 
    the dendritic branch activations at each layer.

    Returns:
       branch_activations: dict {layer_idx: tensor of shape [N, output_dim]}
       all_labels: tensor of shape [N]

    Requirements:
       model.forward_with_branch_outputs(x) -> (final, [layer1_act, layer2_act, ..., soma_act])
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # We'll store each layer's activation in a list, then cat them.
    # branch_activations[layer_idx] = list of [batch_size, layer_output_dim]
    branch_activations = {}
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            final_output, per_layer_acts = model.forward_with_branch_outputs(x_batch)

            # per_layer_acts is a list, length = n_branch_layers + 1 (soma)
            for layer_idx, layer_act in enumerate(per_layer_acts):
                if layer_idx not in branch_activations:
                    branch_activations[layer_idx] = []
                branch_activations[layer_idx].append(layer_act.cpu())

            all_labels.append(y_batch.cpu())

    for layer_idx in branch_activations:
        branch_activations[layer_idx] = torch.cat(branch_activations[layer_idx], dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return branch_activations, all_labels


################################################################################
# 2) SIMPLE MUTUAL INFORMATION
################################################################################
def compute_MI(activations, labels, n_bins=10):
    """
    Estimate the mutual information I(activations; labels).
    Here, we do a 1D approach:
      - if activations is [N], or [N,1], we bin them
      - build joint histogram with labels
      - compute I(X;Y) = sum p(x,y) log [ p(x,y)/(p(x)p(y)) ]

    If activations has shape [N, out_dim] with out_dim>1, 
    we might do a dimension reduction or loop over dims. 
    For simplicity, we'll do mean across dim if >1.
    """
    if activations.ndim > 1 and activations.shape[1] > 1:
        activations = activations.mean(dim=1)

    N = len(activations)
    act_min, act_max = activations.min().item(), activations.max().item()
    if act_min == act_max:
        return 0.0

    edges = torch.linspace(act_min, act_max, n_bins+1)
    bin_indices = torch.bucketize(activations, edges) - 1  # [0..n_bins-1]

    num_classes = int(labels.max().item() + 1)
    joint_hist = torch.zeros((n_bins, num_classes), dtype=torch.float)
    for b, c in zip(bin_indices, labels):
        joint_hist[b, c] += 1

    joint_prob = joint_hist / joint_hist.sum()
    act_prob = joint_prob.sum(dim=1, keepdim=True)   # shape [n_bins,1]
    label_prob = joint_prob.sum(dim=0, keepdim=True) # shape [1,num_classes]

    eps = 1e-12
    ratio = (joint_prob+eps) / ((act_prob+eps)*(label_prob+eps))
    mi = (joint_prob * ratio.log2()).sum()
    return mi.item()

################################################################################
# 3) COMPUTE BRANCH INFO
################################################################################
def analyze_branch_information(model, dataset, 
                               batch_size=64, device='cpu', n_bins=10):
    """
    1) Gathers branch activations
    2) For each branch (layer_idx), compute MI with the label
    3) Return dict {layer_idx: MI_value}
    """
    branch_acts, labels = record_branch_activations(model, dataset, batch_size, device)
    branch_info = {}
    for layer_idx, act in branch_acts.items():
        mi_val = compute_MI(act, labels, n_bins=n_bins)
        branch_info[layer_idx] = mi_val
    return branch_info


################################################################################
# 4) ABLATION
################################################################################
def measure_performance(model, dataset, batch_size=64, device='cpu'):
    """
    Evaluate model's classification accuracy (or NLL) on the dataset.
    Modify as needed for your model's output.
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)  # if EINet -> could be distribution or logits
            # if it's distribution with .logits
            if hasattr(output, 'logits'):
                preds = output.logits.argmax(dim=-1)
            else:
                preds = output.argmax(dim=-1)

            total_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_correct / total


def ablate_branch(model, layer_idx, branch_ids=None, mode='direct'):
    """
    Zero out excitatory/inhibitory weights for branch(es) at layer_idx.
    - If branch_ids is None, ablate ALL dendrites at this layer.
    - If branch_ids is a list of indices, ablate only those branches.
    - 'mode=direct' zeros only excit/inhib pre_w, not blockLinear or upstream.

    This ensures we don't remove signals from previous layers, so deeper layers
    can still matter after ablating a lower layer.

    We need to adjust to match DendriticBranchLayer structure. Typically:
      model.branch_layers[layer_idx] -> has .branch_excitation and .branch_inhibition
    """
    layer = model.branch_layers[layer_idx]
    
    if branch_ids is None:
        if layer.input_excitatory:
            layer.branch_excitation.pre_w.data.zero_()
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            layer.branch_inhibition.pre_w.data.zero_()

    elif mode == 'direct':
        if layer.input_excitatory:
            for b_id in branch_ids:
                layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            for b_id in branch_ids:
                layer.branch_inhibition.pre_w.data[b_id, :] = 0.

    elif mode == 'full':
        if layer.input_excitatory:
            if branch_ids is None:
                layer.branch_excitation.pre_w.data.zero_()
            else:
                for b_id in branch_ids:
                    layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            if branch_ids is None:
                layer.branch_inhibition.pre_w.data.zero_()
            else:
                for b_id in branch_ids:
                    layer.branch_inhibition.pre_w.data[b_id, :] = 0.
        # We do NOT zero out blockLinear or anything that kills upstream input.
        # That way deeper layers still get signals from earlier layers.


def analyze_branch_ablation(model, dataset, layer_idx, branch_ids,
                            mode='direct', device='cpu'):
    """
    1) measure baseline accuracy
    2) for each branch_id in branch_ids, ablate, measure accuracy drop
    3) restore weights

    If we pass branch_ids=None, it ablates the entire layer. 
    """
    baseline_acc = measure_performance(model, dataset, device=device)
    results = {}
    original = {}
    for n, p in model.named_parameters():
        original[n] = p.detach().clone()

    # If branch_ids is None, we do a single ablation of the whole layer
    # If it's a list, we do one ablation per branch in that list
    if branch_ids is None:
        ablate_branch(model, layer_idx, branch_ids=None, mode=mode)
        acc_after = measure_performance(model, dataset, device=device)
        delta = baseline_acc - acc_after
        results["all_branches"] = (acc_after, delta)

        for n, p in model.named_parameters():
            p.data.copy_(original[n])
    else:
        # ablate each branch ID separately
        for b_id in branch_ids:
            ablate_branch(model, layer_idx, [b_id], mode=mode)
            acc_after = measure_performance(model, dataset, device=device)
            delta = baseline_acc - acc_after
            results[b_id] = (acc_after, delta)

            # restore
            for n, p in model.named_parameters():
                p.data.copy_(original[n])
    
    return baseline_acc, results





def compute_entropy(prob_dist):
    """
    Given a discrete probability distribution (torch tensor),
    compute the Shannon entropy: H = - sum(p * log2 p).
    """
    prob_dist = prob_dist.clamp(min=1e-12)
    return -(prob_dist * prob_dist.log2()).sum(dim=-1)

def compute_mutual_information(joint_prob, marginal_x, marginal_y):
    """
    MI(X,Y) = sum_{x,y} p(x,y) log( p(x,y) / [p(x)p(y)] ).
    All inputs are torch tensors of probabilities.
    """
    joint_prob = joint_prob.clamp(min=1e-12)
    marginal_x = marginal_x.clamp(min=1e-12)
    marginal_y = marginal_y.clamp(min=1e-12)
    
    ratio = joint_prob / (marginal_x[:,None] * marginal_y[None,:])
    return (joint_prob * ratio.log2()).sum()



def estimate_hidden_output_MI(model, data_loader, n_bins=10):
    """
    Discretize a hidden layer's activation into n_bins and estimate
    mutual information with the class label. This is a rough approximation:
    1. Collect hidden activation for each sample & label
    2. Bin them to get histogram p(hidden_bin, label)
    3. Compare to p(hidden_bin)*p(label)
    """
    model.eval()
    hidden_acts = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            # forward pass
            # let's extract the second-to-last layer's output or any hidden layer
            # For demonstration, assume "model.net.layers[-1].excitatory_cells"
            # or we adapt the model forward pass to return hidden acts
            activations = model.net.layers[-1].excitatory_cells.forward(x)
            hidden_acts.append(activations.cpu())
            labels.append(y.cpu())

    hidden_acts = torch.cat(hidden_acts, dim=0)  # shape [N, n_excitatory]
    labels = torch.cat(labels, dim=0)           # shape [N]

    # We do this for a single dimension or average across dimensions
    # for simplicity, let's just do dimension 0:
    h0 = hidden_acts[:, 0]
    min_val, max_val = h0.min(), h0.max()

    bin_edges = torch.linspace(min_val, max_val, n_bins+1)
    bin_indices = torch.bucketize(h0, bin_edges) - 1  # in [0, n_bins-1]
    

    n_classes = labels.max().item() + 1
    joint_hist = torch.zeros((n_bins, n_classes), dtype=torch.float)
    for b, c in zip(bin_indices, labels):
        joint_hist[b, c] += 1
    joint_prob = joint_hist / joint_hist.sum()

    marginal_hidden = joint_prob.sum(dim=1)  # sum across classes
    marginal_label = joint_prob.sum(dim=0)   # sum across bins


    mi_est = compute_mutual_information(joint_prob, marginal_hidden, marginal_label)
    return mi_est.item()