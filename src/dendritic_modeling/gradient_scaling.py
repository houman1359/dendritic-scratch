"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement 
various gradient-scaling strategies for BlockLinear or other modules.
"""

import torch


def scale_activation_grad_hook(module, grad_input, grad_output):
    return grad_output


def scale_layer_gradients_elementwise(module, grad_input, grad_output, scale_vec):
    return tuple(
        grad * scale_vec if grad is not None else None for grad in grad_input
    )


def scale_layer_gradients_scalar(module, grad_input, grad_output, scale_factor):
    """
    Multiply entire grad_input by a single scalar factor.
    """
    if grad_input is None:
        return None
    return tuple(
        g * scale_factor if (g is not None) else None
        for g in grad_input
    )

class GradientScaler:
    """
    A class for hooking into BlockLinear or other modules to apply
    user-selected gradient scaling strategies.
    """

    def __init__(self,reactivation_strategy="none", blocklinear_strategy="none", scale_factor=1.0, layer_idx=0):
        """
        Parameters
        ----------
        strategy : str
            One of ['none', 'distal_upweight_by_idx', 'block_conductance_dynamic'].
        scale_factor : float
            Base scaling factor for certain strategies.
        layer_idx : int
            For 'distal_upweight_by_idx' if needed.
        """
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx
    
    def register_reactivation_inverse(self, reactivation_layer):
        """
        Register a backward hook to invert the reactivation function gradient.
        """
        if self.reactivation_strategy != 'none':
            reactivation_layer.register_full_backward_hook(scale_activation_grad_hook)


    def register_block_linear_dynamic(self, block_linear_layer, branch_excitation=None, branch_inhibition=None):
        """
        If strategy='block_conductance_dynamic', attach a backward hook that 
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """

        def hook_fn(module, grad_input, grad_output):
            if self.blocklinear_strategy == 'block_conductance_dynamic':
                scale_vec = compute_block_linear_grad_scale(
                    block_linear_layer=block_linear_layer,
                    branch_excitation=branch_excitation,
                    branch_inhibition=branch_inhibition
                )
                return scale_layer_gradients_elementwise(module, grad_input, grad_output, scale_vec)

            elif self.blocklinear_strategy == 'distal_upweight_by_idx':
                scale_val = (self.layer_idx + 1) * self.scale_factor
                return scale_layer_gradients_scalar(module, grad_input, grad_output, scale_val)

            else:
                return grad_input

        if self.blocklinear_strategy != 'none':
            block_linear_layer.register_full_backward_hook(hook_fn)


def compute_block_linear_grad_scale(block_linear_layer, branch_excitation=None, branch_inhibition=None):
    """
    A conduction-based dynamic scale, returning a flattened vector so each weight 
    can get a distinct multiplier. shape [out_features * block_size].
    """
    with torch.no_grad():
        total_conductance = block_linear_layer.sum_conductances() + 1.0

        if branch_excitation is not None:
            total_conductance += branch_excitation.pruned_weight().sum(dim=1)
        if branch_inhibition is not None:
            total_conductance += branch_inhibition.pruned_weight().sum(dim=1)

        block_exp = block_linear_layer.log_weight.exp()
        ratio_2d = total_conductance[:, None] / block_exp  # shape [out_features, block_size]
        ratio_1d = ratio_2d.flatten()                      # shape [out_features * block_size]
    return ratio_1d