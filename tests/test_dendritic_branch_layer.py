import unittest

import torch
import torch.nn as nn

from dendritic_modeling.dendrinet import (TopKLinear, 
                                          BlockLinear, 
                                          DendriticBranchLayer)


class TestDendriticBranchLayer(unittest.TestCase):

    def setUp(self):
        # Set dimensions and parameters for test cases
        self.output_dim = 5
        self.excitatory_input_dim = 10
        self.excitatory_synapses_per_branch = 3
        self.inhibitory_input_dim = 10
        self.inhibitory_synapses_per_branch = 3
        self.input_branch_factor = 2

    def test_initialization_no_inhibition_no_branch(self):
        # Test DendriticBranchLayer without inhibitory input and branch input
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     reactivate=False)
        self.assertTrue(isinstance(layer.branch_excitation, TopKLinear))
        self.assertFalse(hasattr(layer, 'branch_inhibition'))
        self.assertFalse(hasattr(layer, 'branches_to_output'))
        self.assertFalse(layer.reactivate)

    def test_initialization_with_inhibition_and_branch(self):
        # Test DendriticBranchLayer with inhibitory and branch inputs
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     self.inhibitory_input_dim,
                                     self.inhibitory_synapses_per_branch,
                                     self.input_branch_factor,
                                     reactivate=True)
        
        self.assertTrue(isinstance(layer.branch_excitation, TopKLinear))
        self.assertTrue(isinstance(layer.branch_inhibition, TopKLinear))
        self.assertTrue(isinstance(layer.branches_to_output, BlockLinear))
        self.assertTrue(layer.reactivate)

    def test_forward_no_inhibition_no_branch(self):
        # Forward pass without inhibition and branch inputs
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch)
        x = torch.randn(2, self.excitatory_input_dim)  # Batch size of 2
        output = layer(x)
        self.assertEqual(output.shape, (2, self.output_dim))

    def test_forward_with_inhibition_and_branch(self):
        # Forward pass with inhibition and branch inputs
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     self.inhibitory_input_dim,
                                     self.inhibitory_synapses_per_branch,
                                     self.input_branch_factor)
        x = torch.randn(2, self.excitatory_input_dim)  # Batch size of 2
        inhibitory_input = torch.randn(2, self.inhibitory_input_dim)
        branch_input = torch.randn(2, 
                                   self.output_dim * self.input_branch_factor)
        output = layer(x, inhibitory_input, branch_input)
        self.assertEqual(output.shape, (2, self.output_dim))

    def test_forward_with_reactivation(self):
        # Test reactivation in forward pass
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     reactivate=True)
        x = torch.randn(2, self.excitatory_input_dim)  # Batch size of 2
        output = layer(x)
        self.assertEqual(output.shape, (2, self.output_dim))
        # Check if reactivation threshold applied
         # Should be non-negative due to reactivation
        self.assertTrue(torch.all(output >= 0)) 

    def test_get_weights(self):
        # Check weights retrieval
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     self.inhibitory_input_dim,
                                     self.inhibitory_synapses_per_branch)
        weights = layer.get_weights()
        # Contains both excitatory and inhibitory weights
        self.assertEqual(len(weights), 2) 
        self.assertEqual(weights[0].shape, (self.output_dim, 
                                            self.excitatory_input_dim))
        self.assertEqual(weights[1].shape, (self.output_dim, 
                                            self.inhibitory_input_dim))

    def test_get_mask(self):
        # Check weight mask retrieval
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     self.inhibitory_input_dim,
                                     self.inhibitory_synapses_per_branch)
        masks = layer.get_mask()
        # Contains both excitatory and inhibitory masks
        self.assertEqual(len(masks), 2)  
        self.assertEqual(masks[0].shape, (self.output_dim, 
                                          self.excitatory_input_dim))
        self.assertEqual(masks[1].shape, (self.output_dim, 
                                          self.inhibitory_input_dim))

    def test_get_pruned_weights(self):
        # Check pruned weights retrieval
        layer = DendriticBranchLayer(self.output_dim, 
                                     self.excitatory_input_dim, 
                                     self.excitatory_synapses_per_branch,
                                     self.inhibitory_input_dim,
                                     self.inhibitory_synapses_per_branch)
        pruned_weights = layer.get_pruned_weights()
        # Contains both excitatory and inhibitory pruned weights
        self.assertEqual(len(pruned_weights), 2)  
        self.assertEqual(pruned_weights[0].shape, (self.output_dim, 
                                                   self.excitatory_input_dim))
        self.assertEqual(pruned_weights[1].shape, (self.output_dim, 
                                                   self.inhibitory_input_dim))

if __name__ == "__main__":
    unittest.main()