"""
training.py
===========
This module contains the TrainerMLE class for training the models.
"""

import os
import json
import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from dendritic_modeling import logger


class TrainerMLE(object):
    """
    Maximum Likelihood Estimation (MLE) trainer for training probabilistic 
    models. 

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters during training.

    suppress_prints : bool, optional
        Whether to suppress print statements during training. Default is False.

    print_every : int, optional
        The frequency with which to print training updates. Default is 10.

    Methods
    -------

    train(model, train_data, valid_data, grad_clip_value=5, epochs=100,
            batch_size=256, shuffle=True, plot_losses=False, save_path=None)
            Trains the model on the training data and validates on the 
            validation data.

    Notes
    -----
      - The model must implement a `log_prob` method that computes the log 
        probability of observing the labels given the inputs.
      - train_data: Data object containing training data.
      - valid_data: Data object containing validation data.


    """
    
    def __init__(self, optimizer, suppress_prints = False, print_every = 10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every
    

    def train(
            self, 
            model, 
            train_data, 
            valid_data, 
            grad_clip_value = 5,
            epochs = 100, 
            batch_size = 256, 
            shuffle = True, 
            plot_losses = False, 
            save_path = None,
            ):
        """
        Trains a model using maximum likelihood estimation.

        Parameters
        ----------

        model : ProbabilisticModel
            The model to be trained. Must implement a `log_prob` method that 
            computes the log probability of observing the labels given the 
            inputs.

        train_data : Data
            The training data.

        valid_data : Data
            The validation data.

        grad_clip_value : float, optional
            The value at which to clip the gradients to prevent exploding 
            gradients. The gradients are clipped in the range [-grad_clip_value,
            grad_clip_value]. Default is 5.

        epochs : int, optional
            The number of epochs to train the model. Default is 100.

        batch_size : int, optional
            The batch size used during training. Default is 256.

        shuffle : bool, optional    
            Whether to shuffle the data during training. Default is True.

        plot_losses : bool, optional
            Whether to plot the training and validation losses. 
            Default is False.

        save_path : str, optional
            Directory to save checkpoints and loss curves if provided.

        Returns
        -------

        dict
            A dictionary containing the best epoch, training losses, and
            validation losses. If `plot_losses` is True, the dictionary will
            also contain a figure object for the loss curves and the figure
            will be saved to `save_path` if provided.

        """


        # whether to save at checkpoints
        save_boolean = save_path is not None and os.path.exists(save_path)
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_dataloader = DataLoader(train_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        valid_dataloader = DataLoader(valid_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        best_state_dict = deepcopy(model.state_dict())
        
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(device))
                    label_batch = ag.Variable(label_batch.to(device))
                    
                    # minimize negative log likelihood
                    nll = -1 * model.log_prob(input_batch, 
                                              label_batch).mean(dim = 0)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    nll.backward()
                    # prevent exploding gradients
                    clip_grad_value_(model.parameters(), 
                                     clip_value = grad_clip_value)
                    # update weights
                    self.optimizer.step()
                    # aggregate training loss
                    train_loss += nll.item()
                
                except:
                    # if NaNs encountered in gradients
                    if not self.suppress:
                        logger.info('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        nll = -1 * model.log_prob(
                            input_batch.to(device), 
                            label_batch.to(device)).mean(dim = 0)
                        # compute and aggregate validation loss 
                        valid_loss += nll.item()
                    
                    except:
                        if not self.suppress:
                            logger.info('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())

                if save_boolean:
                    torch.save(best_state_dict, save_path+'/state_dict.pt')

                    with open(save_path+'/losses.json', 'w') as f:
                        json.dump(
                            {'best epoch' : best_epoch, 
                             'train losses' : train_losses, 
                             'valid losses': valid_losses}, 
                             f
                        )

            
            if not self.suppress:
                # printing
                if epoch % self.print_every == 0:
                    logger.info(f'------------------{epoch}------------------')
                    logger.info('training loss: %.2f | validation loss: %.2f' % (
                        train_loss, valid_loss))
                    
                    time_elapsed = (time.time() - stime)
                    pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
                    
                    logger.info('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                        time_elapsed, pred_time_remaining))
        
        # load best model
        model.load_state_dict(best_state_dict)
        
        results = {
            'best epoch' : best_epoch, 
            'train losses' : train_losses, 
            'valid losses' : valid_losses,
            }
        
        if save_boolean:
            with open(save_path+'/losses.json', 'w') as f:
                json.dump(results, f)

        # plot loss curves
        if plot_losses:
            
            Train_losses = np.array(train_losses)
            Train_losses[np.where(Train_losses > 5)] = 5
            
            Valid_losses = np.array(valid_losses)
            Valid_losses[np.where(Valid_losses > 5)] = 5
            
            fig = plt.figure()
            
            plt.plot(range(1,epochs+1), Train_losses, '0.4', label = 'training')
            plt.plot(range(1,epochs+1), Valid_losses, 'b', label = 'validation')
            
            plt.xlabel('Epochs')
            plt.xticks(range(0,epochs+1,int(epochs // 10)))
            plt.ylabel('Negative Log Likelihood')
            plt.title('Loss Curves')
            plt.legend()

            results['loss curves'] = fig

            if save_boolean:
                fig.savefig(save_path+'/loss_curves.jpeg')
                plt.close(fig)
            
        return results
    

# work in progress
class TrainerMLE_EINet_video(object):
    
    def __init__(self, optimizer, suppress_prints = False, print_every = 10):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every
    
    # method for training
    def train(
            self, model, train_data, valid_data, grad_clip_value = 5,
            epochs = 100, batch_size = 256, shuffle = True, 
            plot_losses = False, save_path = None,
            ):
        # model: initialized model to be trained; must implement a .log_prob() function which
        #   produces the log probability of observing the labels given the inputs. The model
        #   must also have a .net attribute which is an instance of the EINet class.
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data

        log = {}
        for l in range(len(model.net.layers)):
            layer = model.net.layers[l]
            inh_dendrinet = layer.inhibitory_cells
            exc_dendrinet = layer.excitatory_cells

            log[f'layer{l}'] = {}
            log[f'layer{l}']['inh_cells'] = {}
            log[f'layer{l}']['exc_cells'] = {}

            if inh_dendrinet.input_inhibitory:
                nbl = inh_dendrinet.n_branch_layers
                for b in range(inh_dendrinet.n_branch_layers):
                    log[f'ei_layer{l}']['inh_cells'][f'br_layer{nbl-b}'] = []
                log[f'layer{l}']['inh_cells']['s_layer'] = []
            
            nbl = exc_dendrinet.n_branch_layers
            for b in range(exc_dendrinet.n_branch_layers):
                log[f'ei_layer{l}']['exc_cells'][f'br_layer{nbl-b}'] = []
            log[f'layer{l}']['exc_cells']['s_layer'] = []

        # whether to save at checkpoints
        save_boolean = save_path is not None and os.path.exists(save_path)
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_dataloader = DataLoader(train_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        valid_dataloader = DataLoader(valid_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        best_state_dict = deepcopy(model.state_dict())
        
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(device))
                    label_batch = ag.Variable(label_batch.to(device))
                    
                    # minimize negative log likelihood
                    nll = -1 * model.log_prob(input_batch, 
                                              label_batch).mean(dim = 0)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    nll.backward()
                    # prevent exploding gradients
                    clip_grad_value_(model.parameters(), 
                                     clip_value = grad_clip_value)
                    # update weights
                    self.optimizer.step()
                    # aggregate training loss
                    train_loss += nll.item()
                
                except:
                    # if NaNs encountered in gradients
                    if not self.suppress:
                        logger.info('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        nll = -1 * model.log_prob(
                            input_batch.to(device), 
                            label_batch.to(device)).mean(dim = 0)
                        # compute and aggregate validation loss 
                        valid_loss += nll.item()
                    
                    except:
                        if not self.suppress:
                            logger.info('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())

                if save_boolean:
                    torch.save(best_state_dict, 
                               save_path+'/state_dict.pt')

                    with open(save_path+'/losses.json', 'w') as f:
                        json.dump(
                            {'best epoch' : best_epoch, 
                             'train losses' : train_losses, 
                             'valid losses': valid_losses}, 
                             f
                        )

            
            if not self.suppress:
                # printing
                if epoch % self.print_every == 0:
                    logger.info(f'------------------{epoch}------------------')
                    logger.info('training loss: %.2f | validation loss: %.2f' % (
                        train_loss, valid_loss))
                    
                    time_elapsed = (time.time() - stime)
                    pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
                    
                    logger.info('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                        time_elapsed, pred_time_remaining))
        
        # load best model
        model.load_state_dict(best_state_dict)
        
        results = {
            'best epoch' : best_epoch, 
            'train losses' : train_losses, 
            'valid losses' : valid_losses,
            }
        
        if save_boolean:
            with open(save_path+'/losses.json', 'w') as f:
                json.dump(results, f)

        # plot loss curves
        if plot_losses:
            
            Train_losses = np.array(train_losses)
            Train_losses[np.where(Train_losses > 5)] = 5
            
            Valid_losses = np.array(valid_losses)
            Valid_losses[np.where(Valid_losses > 5)] = 5
            
            fig = plt.figure()
            
            plt.plot(range(1,epochs+1), 
                     Train_losses, 
                     '0.4', 
                     label = 'training')
            plt.plot(range(1,epochs+1), 
                     Valid_losses, 
                     'b', 
                     label = 'validation')
            
            plt.xlabel('Epochs')
            plt.xticks(range(0,epochs+1,int(epochs // 10)))
            plt.ylabel('Negative Log Likelihood')
            plt.title('Loss Curves')
            plt.legend()

            results['loss curves'] = fig

            if save_boolean:
                fig.savefig(save_path+'/loss_curves.jpeg')
                plt.close(fig)
            
        return results