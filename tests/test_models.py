import unittest

import torch
from torch.distributions import Categorical
from dendritic_modeling.models import (ProbabilisticModel, 
                                       ProbabilisticClassifier, 
                                       ProbabilisticRegressor, 
                                       EINetClassifier)



class ProbabilisticModelTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticModel()

    def test_log_prob(self):
        x = torch.randn(5, 10)  
        y = torch.randint(0, 2, (5,))  
        # test that the method raises a NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model.log_prob(x, y) 

class ProbabilisticClassifierTest(unittest.TestCase):
    def setUp(self):
        class DummyClassifier(ProbabilisticClassifier):
            def forward(self, x):
                # define a binary classifier
                return Categorical(logits=torch.randn(x.size(0), 2)) 
        
        self.model = DummyClassifier()

    def test_predict_deterministic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(((preds == 0) | (preds == 1)).all()) 

    def test_predict_stochastic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=True)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())

class ProbabilisticRegressorTest(unittest.TestCase):
    def setUp(self):
        class DummyRegressor(ProbabilisticRegressor):
            def forward(self, x):
                mean = x.mean(dim=-1, keepdim=True)
                return torch.distributions.Normal(loc=mean, 
                                                  scale=torch.ones_like(mean))
        
        self.model = DummyRegressor()

    def test_predict_deterministic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (5, 1))

    def test_predict_stochastic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=True)
        self.assertEqual(preds.shape, (5, 1))


class EINetClassifierTest(unittest.TestCase):
    def setUp(self):
        # Set up the EINetClassifier with example layer sizes
        self.model = EINetClassifier(
            input_dim=784,
            excitatory_layer_sizes=[10],
            inhibitory_layer_sizes=[20],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[1],
            ee_synapses_per_branch_per_layer=[3],
            ei_synapses_per_branch_per_layer=[3],
            ie_synapses_per_branch_per_layer=[2],
            ii_synapses_per_branch_per_layer=[],
            reactivate=True,
            somatic_synapses=True
        )

    def test_forward_output_shape(self):
        # Batch size of 32, input dim of 784
        x = torch.randn(32, 784)  
        output = self.model.forward(x)
        self.assertIsInstance(output, Categorical)
        # Batch size should be preserved
        self.assertEqual(output.logits.shape[0], 32) 

    def test_predict_deterministic(self):
        x = torch.randn(32, 784)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (32,))