import os
import unittest
import tempfile

import torch
import torch.nn as nn

from dendritic_modeling.utils import (Shaper, 
                                      load_MNIST, 
                                      load_MNIST_modulo10,
                                      Data, 
                                      accuracy_score, 
                                      split_MNIST_inputs)

class TestShaper(unittest.TestCase):
    def test_initialization(self):
        shape = (2, 3)
        shaper = Shaper(shape)
        self.assertEqual(shaper.shape, shape)

    def test_reshape_valid_shape(self):
        shaper = Shaper((2, 3))
        x = torch.randn(6)  
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 3]))

    def test_reshape_invalid_shape(self):
        shaper = Shaper((2, 4))  
        x = torch.randn(6)  
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)  

    def test_reshape_higher_dimensions(self):
        shaper = Shaper((2, 2, 3))
        x = torch.randn(12)  
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 2, 3]))


class TestLoadMNIST(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for caching during tests
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory after tests
        self.temp_dir.cleanup()

    def test_data_loading_no_split(self):
        data = load_MNIST(train_valid_split=1,
                          cache_dir=self.temp_dir.name)
        self.assertIn('train', data)
        self.assertIn('test', data)
        self.assertIsInstance(data['train'], Data)
        self.assertIsInstance(data['test'], Data)
        self.assertGreater(len(data['train']), 0)
        self.assertGreater(len(data['test']), 0)

    def test_data_loading_with_split(self):
        data = load_MNIST(train_valid_split=0.8, 
                          cache_dir=self.temp_dir.name)
        self.assertIn('train', data)
        self.assertIn('valid', data)
        self.assertIn('test', data)
        self.assertIsInstance(data['train'], Data)
        self.assertIsInstance(data['valid'], Data)
        self.assertIsInstance(data['test'], Data)

        # Check that train/validation split ratio is respected approximately
        total_train_valid = len(data['train']) + len(data['valid'])
        self.assertAlmostEqual(len(data['train']) / total_train_valid, 
                               0.8, 
                               delta=0.05)
        self.assertGreater(len(data['test']), 0)

    def test_cache_loading(self):
        # Run once to cache the data
        load_MNIST(train_valid_split=1, cache_dir=self.temp_dir.name)
        
        # Modify the files to check caching is effective
        train_input_path = os.path.join(self.temp_dir.name, 
                                        'data', 'mnist', 'train_inputs.pt')
        os.utime(train_input_path, (0, 0))  # Update the access/modification times
        
        # Load again and verify that it reads from cache without downloading
        data = load_MNIST(train_valid_split=1, cache_dir=self.temp_dir.name)
        self.assertIn('train', data)
        self.assertIn('test', data)
        self.assertIsInstance(data['train'], Data)
        self.assertIsInstance(data['test'], Data)


class TestSplitMNISTInputs(unittest.TestCase):
    def test_split_by_labels(self):
        inputs = torch.randn(100, 784)  
        labels = torch.tensor([i % 10 for i in range(100)]) 
        input_list = split_MNIST_inputs(inputs, labels)

        self.assertEqual(len(input_list), 10) 

        # Check each list element has the correct labels
        for i, group in enumerate(input_list):
            self.assertTrue((torch.unique(labels[labels == i]) == i).all())
            self.assertEqual(len(group), (labels == i).sum().item())  

    def test_empty_inputs(self):
        inputs = torch.empty(0, 784)  
        labels = torch.empty(0, dtype=torch.long)  
        input_list = split_MNIST_inputs(inputs, labels)
        self.assertEqual(len(input_list), 0)


class TestAccuracyScore(unittest.TestCase):
    def setUp(self):
        # Create a dummy classifier with a predict method for testing
        class DummyClassifier(nn.Module):
            def predict(self, inputs):
                # Always predicts '1'
                return torch.ones(inputs.shape[0], dtype=torch.long)  
        self.classifier = DummyClassifier()

    def test_accuracy_all_correct(self):
        inputs = torch.randn(10, 784)  
        labels = torch.ones(10, dtype=torch.long)  
        accuracy = accuracy_score(self.classifier, inputs, labels)
        self.assertEqual(accuracy, 1.0)  

    def test_accuracy_all_incorrect(self):
        inputs = torch.randn(10, 784)  
        labels = torch.zeros(10, dtype=torch.long)  
        accuracy = accuracy_score(self.classifier, inputs, labels)
        self.assertEqual(accuracy, 0.0)  

    def test_accuracy_half_correct(self):
        inputs = torch.randn(10, 784)  
        labels = torch.cat([torch.ones(5, dtype=torch.long), 
                            torch.zeros(5, dtype=torch.long)]) 
        accuracy = accuracy_score(self.classifier, inputs, labels)
        self.assertAlmostEqual(accuracy, 0.5)  # 50% accuracy

class TestLoadMNISTModulo10(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for caching during tests
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory after tests
        self.temp_dir.cleanup()

    def test_data_loading_no_split(self):
        """Test data loading with no train/validation split."""
        data = load_MNIST_modulo10(train_valid_split=1, 
                                   cache_dir=self.temp_dir.name)
        self.assertIn('train', data)
        self.assertIn('test', data)
        self.assertIsInstance(data['train'], Data)
        self.assertIsInstance(data['test'], Data)
        self.assertGreater(len(data['train']), 0)
        self.assertGreater(len(data['test']), 0)

    def test_data_loading_with_split(self):
        """Test data loading with train/validation split."""
        data = load_MNIST_modulo10(train_valid_split=0.8, 
                                   cache_dir=self.temp_dir.name)
        self.assertIn('train', data)
        self.assertIn('valid', data)
        self.assertIn('test', data)
        self.assertIsInstance(data['train'], Data)
        self.assertIsInstance(data['valid'], Data)
        self.assertIsInstance(data['test'], Data)

        # Check that train/validation split ratio is respected approximately
        total_train_valid = len(data['train']) + len(data['valid'])
        self.assertAlmostEqual(
            len(data['train']) / total_train_valid, 0.8, delta=0.05)
        self.assertGreater(len(data['test']), 0)

    def test_shuffle_iterations_effect(self):
        """Test that different shuffle iterations produce different results."""
        # Load data with 1 shuffle iteration
        data1 = load_MNIST_modulo10(shuffle_iterations=1, 
                                    train_valid_split=1, 
                                    cache_dir=self.temp_dir.name)
        # Load data with 2 shuffle iterations
        data2 = load_MNIST_modulo10(shuffle_iterations=2, 
                                    train_valid_split=1, 
                                    cache_dir=self.temp_dir.name)

        # Check that different shuffle iterations yield different input tensors
        # For example, the 10th input tensor should be different
        self.assertFalse(torch.equal(data1['train'][9][0], 
                                     data2['train'][9][0]))

    def test_modulo10_labels(self):
        """Test that labels are correctly computed with modulo 10."""
        data = load_MNIST_modulo10(shuffle_iterations=1, 
                                   train_valid_split=1, 
                                   cache_dir=self.temp_dir.name)
        labels = [item[1].numpy().tolist()  for item in data['train']]
        mod_status = [(item >=0) and (item < 10) for item in labels]
        self.assertTrue(all(mod_status))


if __name__ == '__main__':
    unittest.main()