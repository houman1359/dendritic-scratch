import os
import time
import argparse
from datetime import datetime

import wandb
import torch
import torchinfo
from pathlib import Path

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

from dendritic_modeling.config import load_config



def main(config_path: str, experiment_name: str):

    # Load configuration
    config = load_config(config_path)
    model_hp = config.model.asdict()
    train_hp = config.train.asdict()

    logger.info('loading mnist ...')
    data = load_MNIST(train_valid_split = train_hp['train_valid_split'], 
                      cache_dir=os.getcwd())
    
    # initialize classifier
    clf = EINetClassifierNLL(**model_hp)

    model_summary = torchinfo.summary(
                        clf,
                        input_size=(train_hp['batch_size'], 
                                    model_hp['input_dim']),
                        col_names=["input_size", 
                                   "output_size", 
                                   "num_params"],
                        col_width=25,
                        row_settings=["depth"],
                        depth=7
                    )

    logger.debug("".join(["\n", str(model_summary), "\n"]))

    # initialize trainer class
    trainer = TrainerMLE(
        optimizer = torch.optim.Adam(clf.parameters(), 
                                     lr = train_hp['lr']),
        suppress_prints = False, print_every = 10,
    )

    # create saving directory
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root, 'results', f'{experiment_name}_{now}')
    os.makedirs(save_path, exist_ok = True)


    # intialize W&B
    if config.wandb is not None:
        try:
            wandb_dir = Path(save_path) / "wandb"
            wandb_run_name = experiment_name
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(
                dir=wandb_dir,
                project=config.wandb.project,
                entity=config.wandb.entity,
                group=config.wandb.group,
                name=wandb_run_name,
                tags=config.wandb.tags,
                config=config.asdict(exclude=["wandb"]),
            )
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B ... ")
            config.wandb = None

    print('Training ...\n')
    start_time = time.time()
    trainer.train(
        model=clf,
        train_data=data['train'], 
        valid_data=data['valid'],
        epochs=config.train.epochs, 
        batch_size=train_hp['batch_size'], 
        shuffle=True,
        plot_losses=True, 
        save_path=save_path,
    )
    training_time = time.time() - start_time
    formatted_training_time = f'{training_time:.6f} sec -> {training_time / 60:.6f} min'

    # Evaluate performance
    print('\nEvaluating performance...')
    inference_start = time.time()
    performance = {
        'train accuracy': accuracy_score(clf, *data['train'][:]).item(),
        'valid accuracy': accuracy_score(clf, *data['valid'][:]).item(),
        'test accuracy': accuracy_score(clf, *data['test'][:]).item(),
    }
    inference_time = time.time() - inference_start
    performance['training time'] = formatted_training_time
    performance['inference time'] = f'{inference_time:.6f} sec -> {inference_time / 60:.6f} min'

    # Saving
    save_dict(model_hp, save_path, 'model_hparams.json')
    save_dict(train_hp, save_path, 'training_hparams.json')
    save_dict(performance, save_path, 'performance.json')

    print('  done..\n')

    # Visualize and save model parameters and activations
    shaper = Shaper(shape=(28, 28))
    print('Creating parameter visuals...')
    plot_einet_params(
        einet=clf.net, save_root=os.path.join(save_path, 'param_visuals'),
        reshape_fn=shaper.reshape, reshape_exc_syn=True,
    )

    input_list = split_MNIST_inputs(*data['train'][:])
    print('Creating activation visuals...')
    plot_einet_activations(
        einet=clf.net, input_list=input_list,
        save_root=os.path.join(save_path, 'activation_visuals'),
    )
    print('Done.')

    wandb.finish()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--experiment_name", default="example", help="Optional experiment name for saving results.")
 
    args = parser.parse_args()

    # Run the main function
    main(config_path=args.config, experiment_name=args.experiment_name)

    # Example usage:
    # python scripts/example_with_config.py configs/example_config.yaml --experiment_name example