"""
schemes.py
==========
This module contains the training schemes for the dendritic_modeling package.
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
        #   produces the log probability of observing the labels given the inputs
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        
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

                if save_path is not None and os.path.exists(save_path):
                    torch.save(best_state_dict, save_path+'/state_dict.pt')

                    with open(save_path+'/losses.json', 'w') as f:
                        json.dump(
                            {'train losses' : train_losses, 
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

            return best_epoch, train_losses, valid_losses, fig
            
        return best_epoch, train_losses, valid_losses