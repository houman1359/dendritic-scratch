import os
import time
from datetime import datetime

import torch
import torchinfo

from dendritic_modeling import logger
from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.models import EINetClassifierNLL
from dendritic_modeling.plot_utils import (plot_einet_params, 
                                           plot_einet_activations)
from dendritic_modeling.utils import (load_MNIST, 
                                      split_MNIST_inputs, 
                                      accuracy_score, 
                                      save_dict, 
                                      Shaper)

logger.info('\nimports complete\n')


if __name__ == '__main__':

    name = 'example'
    
    # define model hparams
    model_hp = {
        'input_dim' : 784,
        'excitatory_layer_sizes' : [10,],
        'inhibitory_layer_sizes' : [20,],
        'excitatory_branch_factors' : [2,2,2],
        'inhibitory_branch_factors' : [],
        'ee_synapses_per_branch_per_layer' : [24,],
        'ei_synapses_per_branch_per_layer' : [100,],
        'ie_synapses_per_branch_per_layer' : [1,],
        'ii_synapses_per_branch_per_layer' : [],
        'reactivate' : True,
        'somatic_synapses' : True,
    }

    # define training hparams
    train_hp = {
        'train_valid_split' : 0.8,
        'lr' : 1e-3,
        'epochs' : 300,
        'batch_size' : 1024,
    }

    logger.info('loading mnist ...')
    data = load_MNIST(train_valid_split = train_hp['train_valid_split'], 
                      cache_dir=os.getcwd())

    # initialize classifier
    clf = EINetClassifierNLL(**model_hp)
    model_summary = torchinfo.summary(
                        clf,
                        input_size=(train_hp['batch_size'], model_hp['input_dim']),
                        col_names=["input_size", "output_size", "num_params"],
                        col_width=25,
                        row_settings=["depth"],
                        depth=7
                    )

    logger.debug("".join(["\n", str(model_summary), "\n"]))


    # initialize trainer class (maximum likelihood estimation)
    trainer = TrainerMLE(
        optimizer = torch.optim.Adam(clf.parameters(), lr = train_hp['lr']),
        suppress_prints = False, print_every = 10,
    )

    # create saving directory
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'results', f'{name}_{now}')
    os.makedirs(save_path, exist_ok = True)

    logger.info('training.....\n')
    stime = time.time()
    trainer.train(
        model = clf, 
        train_data = data['train'], valid_data = data['valid'],
        epochs = train_hp['epochs'], batch_size = train_hp['batch_size'], shuffle = True,
        plot_losses = True, save_path = save_path,
        )
    etime = time.time() - stime

    train_time = '%.6f sec -> %.6f min' % (etime, etime/60)
    

    logger.info('\nevaluating performance.....')
    stime = time.time()
    performance = {
        'train accuracy' : accuracy_score(clf, *data['train'][:]).item(),
        'valid accuracy' : accuracy_score(clf, *data['valid'][:]).item(),
        'test accuracy' : accuracy_score(clf, *data['test'][:]).item(),
    }
    etime = time.time() - stime

    performance['training time'] = train_time
    performance['inference time'] = '%.6f sec -> %.6f min' % (etime, etime/60)

    # saving
    save_dict(dict_obj = model_hp, save_path = save_path, fname = 'model_hparams.json')
    save_dict(dict_obj = train_hp, save_path = save_path, fname = 'training_hparams.json')
    save_dict(dict_obj = performance, save_path = save_path, fname = 'performance.json')

    logger.info('  done..\n')

    # initialize shaper class for reshaping weights
    shaper = Shaper(shape = (28,28))

    logger.info('creating param visuals.....')
    plot_einet_params(
        einet = clf.net, save_root = os.path.join(save_path, 'param_visuals'),
        reshape_fn = shaper.reshape, reshape_exc_syn = True, logspace = False,
    )
    logger.info('  done..\n')

    # split inputs into list of tensors
    #   each element in list is tensor of all samples with same label
    input_list = split_MNIST_inputs(*data['train'][:])

    logger.info('creating activation visuals.....')
    plot_einet_activations(
        einet = clf.net, input_list = input_list,
        save_root = os.path.join(save_path, 'activation_visuals'),
    )
    logger.info('  done..\n')
