"""
train_experiments.py
--------------------
Main script that reads config_exp.yaml, builds the model,
chooses a training strategy, trains, and evaluates using
a unified data-loading function from synthetic_datasets.py.
"""
# python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml

import os
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import wandb
import torch
import torchinfo

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, MLPExcInhNetwork)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters.__dict__
    enable_branch_outputs = net_params.pop('enable_branch_outputs', False)
    learning_strategy = net_params.pop('learning_strategy', 'mle')

    if task == 'classification':
        if net_type == 'MLP':
            logger.info(f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = MLPExcInhNetwork(**net_params)
            output_dim = net_params.get('output_dim', 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)

        elif net_type == 'EINet':
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(
                **net_params,
                enable_branch_outputs=enable_branch_outputs,
                learning_strategy=learning_strategy
            )
            output_dim = net_params['excitatory_layer_sizes'][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)

        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Invalid task: {task}")


def main(config_path: str, output_dir: str, experiment_name: str):
    os.makedirs(output_dir, exist_ok=True)
    logger_manager.set_log_file(os.path.join(output_dir, "dendritic_modeling.log"))

    config = load_config(config_path)
    model_cfg = config.model
    train_cfg = config.train
    visual_cfg = config.visualization
    wandb_cfg = config.wandb

    if getattr(wandb_cfg, "use_wandb", False):
        try:
            wandb_dir = Path(output_dir) / "wandb"
            wandb_run_name = experiment_name
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_dict = asdict(config)
            wandb.init(
                dir=str(wandb_dir),
                entity=wandb_cfg.entity,
                project=wandb_cfg.project,
                group=wandb_cfg.group,
                name=wandb_run_name,
                tags=wandb_cfg.tags,
                config=wandb_dict,
            )
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            raise e

    logger.info("Loading dataset...")
    train_ds, valid_ds, test_ds = get_unified_datasets(config.task, train_cfg)

    # Adjust net parameters: input_dim, output_dim
    model_cfg.network.parameters.input_dim = train_ds[0][0].shape[-1]
    model_cfg.network.parameters.output_dim = torch.unique(train_ds[:][1]).shape[-1]
    model_cfg.network.parameters.learning_strategy = getattr(train_cfg, "learning_strategy", "mle")

    clf = initialize_model(model_cfg)

    # We'll store param_count from torchinfo summary in case we want to save it
    param_count = None

    # Attempt to get param_count from summary
    try:
        sample_input, _ = train_ds[0]
        sample_input_dim = sample_input.numel()
        summary = torchinfo.summary(
            clf,
            input_size=(train_cfg.batch_size, sample_input_dim),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7
        )
        logger.debug("\n" + str(summary))
        param_count = summary.total_params

        if wandb.run:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except:
        logger.warning("Could not run torchinfo.summary; param_count will remain None.")

    learning_strategy = getattr(train_cfg, "learning_strategy", "mle")
    logger.info(f"Using learning strategy: {learning_strategy}")

    optimizer = CustomWeightDecayOptimizer(
        model = clf,
        optimizer = torch.optim.Adam(clf.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )
    trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = visual_cfg.save_path
    run_save_path = os.path.join(root, f"{experiment_name}_{now}")
    os.makedirs(run_save_path, exist_ok=True)
    logger.info(f"Saving results and visualizations to: {run_save_path}")

    # We'll store final validation or test "loss" after training, if available
    final_loss_val = None

    if hasattr(visual_cfg, "visualize_epochs") and visual_cfg.visualize_epochs:
        total_epochs = train_cfg.epochs
        visualize_epochs = sorted(visual_cfg.visualize_epochs)
        current_epoch = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Possibly do initial plotting if enabled
        if config.visualization.enabled:
            plot_manager = PlotManager(clf, {"train": train_ds, "valid": valid_ds, "test": test_ds},
                                       config, run_save_path, file_name="_init", device=device)
            for plot_name in config.visualization.plots:
                plot_manager.run_plot(plot_name)

        valid_losses= []
        train_losses= []
        while current_epoch < total_epochs:
            results = trainer.train(
                model=clf,
                train_data=train_ds,
                valid_data=valid_ds,
                epochs=1,
                batch_size=train_cfg.batch_size,
                grad_clip_value=train_cfg.grad_clip_value,
                shuffle=True,
                load_best_state_dict=False,
                plot_losses=False,
                save_path=run_save_path
            )
            valid_losses += results["valid losses"]
            train_losses += results["train losses"]
            current_epoch += 1
            logger.info(f"Completed epoch {current_epoch}/{total_epochs}")

            if current_epoch in visualize_epochs:
                # Evaluate
                evaluate_accuracy(
                    clf, train_ds, valid_ds, test_ds,
                    save_path = os.path.join(run_save_path, 'performance'),
                    filename = f"epoch{current_epoch}",
                )
                # Possibly do more plotting
                if visual_cfg.enabled and visual_cfg.plots:
                    plot_manager = PlotManager(clf, {"train": train_ds, "valid": valid_ds, "test": test_ds},
                                               config, run_save_path, file_name=f"epoch{current_epoch}", device=device)
                    for plot_name in config.visualization.plots:
                        plot_manager.run_plot(plot_name)
        # After final epoch
        if len(valid_losses) > 0:
            final_loss_val = valid_losses[-1]

        plot_NLL_loss_curves(train_losses, valid_losses, current_epoch, run_save_path)

    else:
        # Single-phase or multi-phase training
        if learning_strategy == "freeze_layers":
            epochs_per_layer = getattr(train_cfg, "epochs_per_layer", 5)
            freeze_results = trainer.train_freeze_layers(
                model=clf,
                train_data=train_ds,
                valid_data=valid_ds,
                epochs_per_layer=epochs_per_layer,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                grad_clip_value=train_cfg.grad_clip_value,
                plot_losses=True,
                save_path=run_save_path
            )
            # If we want the final valid loss
            if "valid losses" in freeze_results and len(freeze_results["valid losses"])>0:
                final_loss_val = freeze_results["valid losses"][-1]

        elif learning_strategy in ["two_step", "two_step_kl"]:
            pretrain_epochs = getattr(train_cfg, "pretrain_epochs", 5)
            maintrain_epochs = getattr(train_cfg, "epochs", 45)
            train_results = trainer.train(
                model=clf,
                train_data=train_ds,
                valid_data=valid_ds,
                pretrain_epochs=pretrain_epochs,
                maintrain_epochs=maintrain_epochs,
                batch_size=train_cfg.batch_size,
                grad_clip_value=train_cfg.grad_clip_value,
                shuffle=True,
                load_best_state_dict=True,
                plot_losses=True,
                save_path=run_save_path
            )
            if "valid losses" in train_results and len(train_results["valid losses"])>0:
                final_loss_val = train_results["valid losses"][-1]

        else:
            # plain MLE
            train_results = trainer.train(
                model=clf,
                train_data=train_ds,
                valid_data=valid_ds,
                epochs=train_cfg.epochs,
                batch_size=train_cfg.batch_size,
                grad_clip_value=train_cfg.grad_clip_value,
                shuffle=True,
                load_best_state_dict=True,
                plot_losses=True,
                save_path=run_save_path
            )
            if "valid losses" in train_results and len(train_results["valid losses"])>0:
                final_loss_val = train_results["valid losses"][-1]

    # Evaluate performance on train/valid/test sets
    logger.info("Evaluating performance...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)

    train_acc, valid_acc, test_acc = evaluate_accuracy(
        clf, train_ds, valid_ds, test_ds,
        save_path = os.path.join(run_save_path, 'performance'),
        filename = "final",
    )
    logger.info(f"train acc={train_acc:.3f}, valid acc={valid_acc:.3f}, test acc={test_acc:.3f}")

    if wandb.run is not None:
        wandb.log({"train_acc": train_acc, "valid_acc": valid_acc, "test_acc": test_acc})
        wandb.finish()

    # Possibly do final plotting if visualizations are enabled
    if getattr(config.visualization, "enabled", True) and config.visualization.plots is not None:
        plot_manager = PlotManager(clf, {"train": train_ds, "valid": valid_ds, "test": test_ds},
                                   config, run_save_path, file_name="final", device=device)
        for plot_name in config.visualization.plots:
            plot_manager.run_plot(plot_name)
    else:
        logger.info("Visualization disabled; final plotting skipped.")

    logger.info("All done.")

    # -- ADD THE FINAL JSON DUMP HERE --
    # If no final_loss_val was found, let's fallback to (1.0 - test_acc) or None
    final_loss = final_loss_val if final_loss_val is not None else (1.0 - test_acc)
    final_json_info = {
        "param_count": param_count,
        "final_loss":  float(final_loss) if final_loss is not None else None
    }
    final_json_path = os.path.join(run_save_path, "final_results.json")
    with open(final_json_path, "w") as f:
        json.dump(final_json_info, f, indent=2)
    logger.info(f"Saved final_results.json => {final_json_path}")


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    parser.add_argument("--experiment_name", default="example", help="Experiment name (used for W&B run name).")
    args = parser.parse_args()

    main(config_path=args.config, output_dir=args.output_dir, experiment_name=args.experiment_name)