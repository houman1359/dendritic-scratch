"""
tuning.py
=========
This module contains the hyperparameter tuning class for the dendritic_modeling.
"""

import os
import time

import torch
import matplotlib.pyplot as plt

from dendritic_modeling import logger
from dendritic_modeling.utils import save_dict
from dendritic_modeling.training import TrainerMLE


class DendriNetTuner(object):

    def __init__(
        self, branch_factors_space, n_inh_cells_space, syn_per_inh_cell_space,
        inh_syn_per_branch_space, exc_syn_per_branch_space,
        shunting_space, reactivate_space, b_trainable_space, log_lr_space,
    ):
        hparams = {}
        hparams['branch factors'] = branch_factors_space
        hparams['n inh cells'] = n_inh_cells_space
        hparams['syn per inh cell'] = syn_per_inh_cell_space
        hparams['inh syn per branch'] = inh_syn_per_branch_space
        hparams['exc syn per branch'] = exc_syn_per_branch_space
        hparams['shunting'] = shunting_space
        hparams['reactivate'] = reactivate_space
        hparams['b trainable'] = b_trainable_space
        hparams['log lr'] = log_lr_space

        self.hparams = hparams

        metrics = torch.zeros(
            (3, 
             len(branch_factors_space), 
             len(n_inh_cells_space), 
             len(syn_per_inh_cell_space),
             len(inh_syn_per_branch_space), 
             len(exc_syn_per_branch_space),
             len(shunting_space), 
             len(reactivate_space), 
             len(b_trainable_space), 
             len(log_lr_space)), 
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metrics = metrics.to(device)

        self.n_configs = metrics.flatten(1,-1).shape[-1]
    
    def tune(
        self, dendrinet, input_dim, output_dim,
        train_data, valid_data, test_data,
        epochs, batch_size, metric_fn, save_root,
    ):
        metrics_path = os.path.join(save_root, 'metrics.pt')
        if os.path.exists(metrics_path):
            self.metrics = torch.load(metrics_path, weights_only = True)
        
        config_count = 1
        stime = time.time()
    
        #TODO: Refactor the following nested for loops
        for i in range(len(self.hparams['branch factors'])):
            for j in range(len(self.hparams['n inh cells'])):
                for k in range(len(self.hparams['syn per inh cell'])):
                    for l in range(len(self.hparams['inh syn per branch'])):
                        for m in range(len(self.hparams['exc syn per branch'])):
                            for n in range(len(self.hparams['shunting'])):
                                for o in range(len(self.hparams['reactivate'])):
                                    for p in range(len(self.hparams['b trainable'])):
                                        for q in range(len(self.hparams['log lr'])):

                                            branch_factors = self.hparams['branch factors'][i]
                                            n_inhibitory_cells = self.hparams['n inh cells'][j]
                                            synapses_per_inhibitory_cell = self.hparams['syn per inh cell'][k]
                                            inhibitory_synapses_per_branch = self.hparams['inh syn per branch'][l]
                                            excitatory_synapses_per_branch = self.hparams['exc syn per branch'][m]
                                            shunting = self.hparams['shunting'][n]
                                            reactivate = self.hparams['reactivate'][o]
                                            b_trainable = self.hparams['b trainable'][p]
                                            lr = 10**self.hparams['log lr'][q]

                                            model_dir = f'brf{branch_factors}_nic{n_inhibitory_cells}_sic{synapses_per_inhibitory_cell}_'
                                            model_dir += f'isb{inhibitory_synapses_per_branch}_esb{excitatory_synapses_per_branch}_'
                                            model_dir += f'shu{shunting}_rea{reactivate}_btr{b_trainable}_lr{lr}'

                                            save_path = os.path.join(save_root, 'models', model_dir)

                                            if not os.path.exists(save_path):

                                                logger.info('-------------------------------------')
                                                logger.info(f'model configuration - {config_count}/{self.n_configs}')
                                                logger.info(f'branch factors: {branch_factors}')
                                                logger.info(f'n inhibitory cells: {n_inhibitory_cells}')
                                                logger.info(f'synapses per inhibitory cell: {synapses_per_inhibitory_cell}')
                                                logger.info(f'inhibitory synapses per branch: {inhibitory_synapses_per_branch}')
                                                logger.info(f'excitatory synapses per branch: {excitatory_synapses_per_branch}')
                                                logger.info(f'shunting: {shunting}')
                                                logger.info(f'reactivate: {reactivate}')
                                                logger.info(f'b trainable: {b_trainable}')
                                                logger.info(f'learning rate: {lr}')

                                                
                                                model = dendrinet(
                                                    input_dim = input_dim, output_dim = output_dim,
                                                    branch_factors = branch_factors,
                                                    n_inhibitory_cells = n_inhibitory_cells,
                                                    synapses_per_inhibitory_cell = synapses_per_inhibitory_cell,
                                                    inhibitory_synapses_per_branch = inhibitory_synapses_per_branch,
                                                    excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                                                    shunting = shunting,
                                                    reactivate = reactivate, 
                                                    b_trainable = b_trainable,
                                                )
                                                #scripted_model = torch.jit.script(model)

                                                trainer = TrainerMLE(
                                                    optimizer = torch.optim.Adam(model.parameters(), lr = lr),
                                                    suppress_prints = True,
                                                )
                                                logger.info('training ...')
                                                res = trainer.train(
                                                    model, train_data = train_data, valid_data = valid_data,
                                                    epochs = epochs, batch_size = batch_size, shuffle = True, plot_losses = True,
                                                )
                                                logger.info('Done!')
                                                etime = time.time() - stime
                                                logger.info('time elapsed: %.2f min | %.3f hrs' % (etime/60, etime/3600))

                                                hp = {
                                                    'input_dim' : input_dim,
                                                    'output_dim' : output_dim,
                                                    'branch_factors' : branch_factors,
                                                    'n_inhibitory_cells' : n_inhibitory_cells,
                                                    'synapses_per_inhibitory_cell' : synapses_per_inhibitory_cell,
                                                    'inhibitory_synapses_per_branch' : inhibitory_synapses_per_branch,
                                                    'excitatory_synapses_per_branch' : excitatory_synapses_per_branch,
                                                    'shunting' : shunting,
                                                    'reactivate' : reactivate,
                                                    'b_trainable' : b_trainable,
                                                    'learning rate' : lr,
                                                }

                                                performance = {
                                                    'training' : metric_fn(model, *train_data[:]).item(),
                                                    'validation' : metric_fn(model, *valid_data[:]).item(),
                                                    'testing' : metric_fn(model, *test_data[:]).item(),
                                                }

                                                self.metrics[0,i,j,k,l,m,n,o,p,q] = performance['training']
                                                self.metrics[1,i,j,k,l,m,n,o,p,q] = performance['validation']
                                                self.metrics[2,i,j,k,l,m,n,o,p,q] = performance['testing']

                                                os.makedirs(save_path, exist_ok = True)

                                                torch.save(model.state_dict(), os.path.join(save_path, 'state_dict.pt'))
                                                torch.save(self.metrics, metrics_path)

                                                save_dict(hp, save_path, 'hparams.json')
                                                save_dict(performance, save_path, 'performance.json')
                                                
                                                res['loss curves'].savefig(os.path.join(save_path, 'loss_curves.jpeg'))
                                                plt.close(res['loss curves'])

                                            config_count += 1

        bestix = torch.unravel_index(self.metrics[2,...].argmax(), 
                                     self.metrics[2,...].shape)

        hp_best = {}
        for i, key in enumerate(self.hparams.keys()):
            hp_best[key] = self.hparams[key][bestix[i]]

        save_dict(self.hparams, save_root, 'hparam_space.json')
        save_dict(hp_best, save_root, 'hparam_best.json')

        sweep_path = os.path.join(save_root, 'sweeping_plots')
        os.mkdir(sweep_path)

        logger.info('generating sweeping plots.....')
        self.sweeping_plots(sweep_path)
        logger.info('  done..')


    def sweeping_plots(self, save_path):
        
        train = self.metrics[0,...]
        valid = self.metrics[1,...]
        test = self.metrics[2,...]

        bestix = torch.unravel_index(test.argmax(), test.shape)

        def plot_hp_sweep(hp_name, hp_dim, quantitative = True):
            
            train_values = []
            valid_values = []
            test_values = []

            for i in range(test.shape[hp_dim]):
                ix = list(bestix)
                ix[hp_dim] = i

                train_values.append(train[ix].item())
                valid_values.append(valid[ix].item())
                test_values.append(test[ix].item())

            fig = plt.figure()

            if quantitative:
                plt.plot(self.hparams[hp_name], 
                         train_values, 
                         marker = 'o', 
                         ms = 8, 
                         color = '0.4', 
                         label = 'train')
                plt.plot(self.hparams[hp_name], 
                         valid_values, 
                         marker = 'o', 
                         ms = 8, 
                         color = 'blue', 
                         label = 'valid')
                plt.plot(self.hparams[hp_name], 
                         test_values, marker = 'o', 
                         ms = 8, color = 'orange', label = 'test')
            
            else:
                x = torch.arange(len(self.hparams[hp_name]))
                width = 0.2

                plt.bar(x - width, 
                        train_values, 
                        width = width, 
                        color = '0.4', 
                        label = 'train')
                plt.bar(x, 
                        valid_values, 
                        width = width, 
                        color = 'blue', 
                        label = 'valid')
                plt.bar(x + width, 
                        test_values, 
                        width = width, 
                        color = 'orange', 
                        label = 'test')
                plt.xticks(x, [str(p) for p in self.hparams[hp_name]])
                #plt.tight_layout()
            
            plt.xlabel(hp_name)
            plt.ylabel('Performance')
            plt.title('Effect of Hyperparameter on Performance')

            fig.savefig(os.path.join(save_path, f'{hp_name}_sweep.jpeg'))
            plt.close(fig)


        plot_hp_sweep(hp_name = 'branch factors', 
                      hp_dim = 0, 
                      quantitative = False)
        plot_hp_sweep(hp_name = 'n inh cells', 
                      hp_dim = 1, 
                      quantitative = True)
        plot_hp_sweep(hp_name = 'syn per inh cell', 
                      hp_dim = 2, 
                      quantitative = True)
        plot_hp_sweep(hp_name = 'inh syn per branch', 
                      hp_dim = 3, 
                      quantitative = True)
        plot_hp_sweep(hp_name = 'exc syn per branch', 
                      hp_dim = 4, 
                      quantitative = True)
        plot_hp_sweep(hp_name = 'shunting', 
                      hp_dim = 5, 
                      quantitative = False)
        plot_hp_sweep(hp_name = 'reactivate', 
                      hp_dim = 6, 
                      quantitative = False)
        plot_hp_sweep(hp_name = 'b trainable', 
                      hp_dim = 7, 
                      quantitative = False)
        plot_hp_sweep(hp_name = 'log lr', 
                      hp_dim = 8, 
                      quantitative = True)


        


