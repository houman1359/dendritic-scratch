import os
import shutil
import unittest

import torch

from dendritic_modeling.dendrinet import DendriNet
from dendritic_modeling.tuning import DendriNetTuner

class TestDendriNetTuner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.branch_factors_space = [1, 2]
        cls.n_inh_cells_space = [1, 2]
        cls.syn_per_inh_cell_space = [1, 2]
        cls.inh_syn_per_branch_space = [1]
        cls.exc_syn_per_branch_space = [1]
        cls.shunting_space = [0.5]
        cls.reactivate_space = [True, False]
        cls.b_trainable_space = [True, False]
        cls.log_lr_space = [-3]
        
        # Create a temporary directory to save results
        cls.save_root = "temp_tuning_results"
        os.makedirs(cls.save_root, exist_ok=True)

        # Instantiate DendriNetTuner with test parameter space
        cls.tuner = DendriNetTuner(
            branch_factors_space=cls.branch_factors_space,
            n_inh_cells_space=cls.n_inh_cells_space,
            syn_per_inh_cell_space=cls.syn_per_inh_cell_space,
            inh_syn_per_branch_space=cls.inh_syn_per_branch_space,
            exc_syn_per_branch_space=cls.exc_syn_per_branch_space,
            shunting_space=cls.shunting_space,
            reactivate_space=cls.reactivate_space,
            b_trainable_space=cls.b_trainable_space,
            log_lr_space=cls.log_lr_space
        )

    @classmethod
    def tearDownClass(cls):
        # Remove temporary directory
        shutil.rmtree(cls.save_root)

    def test_hparam_space(self):
        # Check if hyperparameter space is correctly initialized
        self.assertEqual(len(self.tuner.hparams['branch factors']), 
                         len(self.branch_factors_space))
        self.assertEqual(len(self.tuner.hparams['n inh cells']), 
                         len(self.n_inh_cells_space))

    def test_metrics_shape(self):
        # Check if metrics tensor shape matches expected dimensions
        expected_shape = (
            3, 
            len(self.branch_factors_space), 
            len(self.n_inh_cells_space),
            len(self.syn_per_inh_cell_space),
            len(self.inh_syn_per_branch_space),
            len(self.exc_syn_per_branch_space),
            len(self.shunting_space),
            len(self.reactivate_space),
            len(self.b_trainable_space),
            len(self.log_lr_space),
        )
        self.assertEqual(self.tuner.metrics.shape, expected_shape)