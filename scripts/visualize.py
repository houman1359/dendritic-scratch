import os
import time
from datetime import datetime
from copy import deepcopy
import torch
import torchinfo

from dendritic_modeling import logger
from dendritic_modeling.training import CustomWeightDecayOptimizer, TrainerMLE
from dendritic_modeling.models import EINetClassifierNLL
from dendritic_modeling.plot_utils import (
    plot_einet_params, 
    plot_einet_gradients,
    plot_einet_activations,
    plot_einet_profiles,
    einet_activations_to_csv,
    plot_NLL_loss_curves,
)
from dendritic_modeling.utils import (
    load_MNIST,
    load_MNIST_task_switch,                             
    split_MNIST_inputs, 
    accuracy_score, 
    save_dict, 
    Shaper,
)

torch.manual_seed(47)
torch.cuda.manual_seed(47)
torch.cuda.manual_seed_all(47)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger.info('imports complete\n')


def get_and_save_performance_and_visuals(
    clf, 
    data, 
    n_tasks,
    input_list, 
    reshape_fn, 
    reshape_exc_syn, 
    logspace,
    save_path, 
    filename, 
):
    logger.info('evaluating performance.....')
    performance = {
        'train accuracy' : accuracy_score(clf, *data['train'][:]).item(),
        'valid accuracy' : accuracy_score(clf, *data['valid'][:]).item(),
        'test accuracy' : accuracy_score(clf, *data['test'][:]).item(),
    }
    save_dict(
        dict_obj = performance, 
        save_path = os.path.join(save_path, 'performance'), 
        fname = f'{filename}.json',
    )
    logger.info('  done..')

    logger.info('creating profile visuals.....')
    plot_einet_profiles(
        save_root = os.path.join(save_path, 'profiles'),
        model = clf,
        train_data = data['train'],
        valid_data = data['valid'],
        n_tasks = n_tasks,
        logspace = logspace,
        reshape_fn = reshape_fn,
        save_in_dir = True,
        filename = filename,
    )
    logger.info('  done..')

    logger.info('creating param visuals.....')
    plot_einet_params(
        einet = clf.net, 
        save_root = os.path.join(save_path, 'param_visuals'),
        reshape_fn = reshape_fn, 
        reshape_exc_syn = reshape_exc_syn, 
        logspace = logspace,
        save_in_dir = True,
        filename = filename,
    )
    logger.info('  done..')

    logger.info('creating gradient visuals.....')
    train_inputs, train_labels = data['train'][:]
    plot_einet_gradients(
        model = clf,
        inputs = train_inputs,
        labels = train_labels,
        save_root = os.path.join(save_path, 'grad_visuals'),
        reshape_fn = reshape_fn,
        reshape_exc_syn = reshape_exc_syn,
        save_in_dir = True,
        filename = filename,
    )
    logger.info('  done..')

    # logger.info('creating activation visuals.....')
    # plot_einet_activations(
    #     einet = clf.net, 
    #     input_list = input_list,
    #     save_root = os.path.join(save_path, 'activation_visuals'),
    #     save_in_dir = True,
    #     filename = filename,
    # )
    # logger.info('  done..')

    # logger.info('creating activation component tables.....')
    # einet_activations_to_csv(
    #     einet = clf.net,
    #     input_list = input_list,
    #     save_root = os.path.join(save_path, 'activation_components'),
    #     filename = filename,
    # )
    # logger.info('  done..\n')


if __name__ == '__main__':

    name = 'visualize'
    
    # define model hparams
    model_hp = {
        'input_dim' : 784,
        'excitatory_layer_sizes' : [10,],
        'inhibitory_layer_sizes' : [20,],
        'excitatory_branch_factors' : [2,2],
        'inhibitory_branch_factors' : [],
        'ee_synapses_per_branch_per_layer' : [24,],
        'ei_synapses_per_branch_per_layer' : [100,],
        'ie_synapses_per_branch_per_layer' : [2,],
        'ii_synapses_per_branch_per_layer' : [],
        'reactivate' : True,
        'somatic_synapses' : True,
    }

    # define training hparams
    visualize_hp = {
        'train_valid_split' : 0.8,
        'n_tasks' : 1,
        'lr' : 1e-3,
        'total_epochs' : 300,
        'plot_epochs' : [10, 100, 300],
        'batch_size' : 1024,
        'logspace' : True,
        'weight_decay' : 0.1,
    }

    logger.info('loading mnist ...')
    data = load_MNIST(
        train_valid_split = visualize_hp['train_valid_split'], 
        cache_dir = os.getcwd(),
    )
    # data = load_MNIST_task_switch(
    #     train_valid_split = visualize_hp['train_valid_split'],
    #     cache_dir = os.getcwd(),
    # )
    # split inputs into list of tensors
    #   each element in list is tensor of all samples with same label
    input_list = split_MNIST_inputs(*data['valid'][:])

    # initialize shaper class for reshaping weights
    shaper = Shaper(shape = (28,28))

    # initialize classifier
    clf = EINetClassifierNLL(**model_hp)
    model_summary = torchinfo.summary(
                        clf,
                        input_size=(visualize_hp['batch_size'], model_hp['input_dim']),
                        col_names=["input_size", "output_size", "num_params"],
                        col_width=25,
                        row_settings=["depth"],
                        depth=7
                    )

    logger.debug("".join(["\n", str(model_summary), "\n"]))

    # create saving directory
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'results', f'{name}_{now}')
    os.makedirs(save_path, exist_ok = True)

    get_and_save_performance_and_visuals(
        clf = clf,
        data = data,
        n_tasks = visualize_hp['n_tasks'],
        input_list = input_list,
        reshape_fn = shaper.reshape,
        reshape_exc_syn = True,
        logspace = visualize_hp['logspace'],
        save_path = save_path,
        filename = '_init',
    )

    optimizer = CustomWeightDecayOptimizer(
        model = clf,
        optimizer = torch.optim.Adam(clf.parameters(), lr = visualize_hp['lr']),
        weight_decay = visualize_hp['weight_decay'],
    )

    # initialize trainer class (maximum likelihood estimation)
    trainer = TrainerMLE(
        optimizer = optimizer,
        suppress_prints = True,
    )

    train_losses = []
    valid_losses = []

    plot_epochs = [0] + deepcopy(visualize_hp['plot_epochs'])
    for i in range(1, len(plot_epochs)):
        logger.info(f'training until epoch {plot_epochs[i]} ...')
        results = trainer.train(
            model = clf,
            train_data = data['train'],
            valid_data = data['valid'],
            grad_clip_value = 5,
            epochs = plot_epochs[i] - plot_epochs[i-1],
            batch_size = visualize_hp['batch_size'],
            load_best_state_dict = False,
            shuffle = True,
            plot_losses = False,
        )
        
        train_losses += results['train losses']
        valid_losses += results['valid losses']
        logger.info('  done..')

        get_and_save_performance_and_visuals(
            clf = clf,
            data = data,
            n_tasks = visualize_hp['n_tasks'],
            input_list = input_list,
            reshape_fn = shaper.reshape,
            reshape_exc_syn = True,
            logspace = visualize_hp['logspace'],
            save_path = save_path,
            filename = f'epoch{plot_epochs[i]}',
        )

    fig = plot_NLL_loss_curves(
        train_losses, 
        valid_losses, 
        visualize_hp['total_epochs'],
        save_path = save_path,
    )

    save_dict(
        dict_obj = model_hp, 
        save_path = save_path, 
        fname = 'model_hparams.json',
    )
    save_dict(
        dict_obj = visualize_hp, 
        save_path = save_path, 
        fname = 'visualize_hparams.json',
    )
