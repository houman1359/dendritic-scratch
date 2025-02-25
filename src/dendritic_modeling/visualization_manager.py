"""
visualization_manager.py
========================
Contains a PlotManager class that holds references to all visualization
and analysis functions to produce actual visualizations for "gradients", 
"branch_info", and "ablation" using plot_utils.py functions 
"""

import os
import logging

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dendritic_modeling.synthetic_datasets import split_MNIST_inputs
from dendritic_modeling.utils import (
    Shaper,
    analyze_branch_ablation,
    analyze_branch_information,
    measure_performance
)
from dendritic_modeling.plot_utils import (
    plot_einet_params, 
    plot_einet_activations,
    plot_einet_gradients,
    plot_einet_profiles,
    einet_activations_to_csv,
    plot_NLL_loss_curves,
    plot_depth_info
)

class PlotManager:
    """
    A manager that stores all known plot/analysis methods in a dictionary.
    The main experiment code can call PlotManager.run_plot(plot_name),
    without needing to manually code each function call in train_experiments.py.
    """
    def __init__(self, model, data, config, save_path, file_name, device='cpu'):

        self.model = model
        self.data = data
        self.config = config
        self.device = device

        vis_cfg = self.config.visualization
        self.save_path = save_path
        self.file_name = file_name
        os.makedirs(self.save_path, exist_ok=True)

        self.plot_methods = {
            "weights": self._plot_weights,
            "activations": self._plot_activations,
            "gradients": self._plot_gradients,
            "profiles": self._plot_profiles,
            "branch_info": self._branch_info,
            "ablation": self._ablation,
        }

    def run_plot(self, plot_name, epoch=None):

        if epoch is not None:
            logging.info(f"At epoch {epoch}: generating plot '{plot_name}'...")
        else:
            logging.info(f"Generating plot '{plot_name}'...")
        func = self.plot_methods.get(plot_name, None)
        if func is not None:
            func()
            if epoch is not None:
                logging.info(f"Plot '{plot_name}' saved for epoch {epoch}.")
            else:
                logging.info(f"Plot '{plot_name}' saved.")
        else:
            logging.info(f"Plot name '{plot_name}' not recognized in PlotManager.")

    def _plot_weights(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "param_visuals")
            os.makedirs(save_dir, exist_ok=True)
            shaper = Shaper(shape=(28, 28))
            plot_einet_params(
                einet=self.model.net,
                save_root=save_dir,
                reshape_fn=shaper.reshape,
                reshape_exc_syn=True,
                logspace=True,
                save_in_dir=True,
                filename=self.file_name
            )
        else:
            logging.info("No 'net' attribute found on model. Skipping _plot_weights.")

    def _plot_activations(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "activation_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data['train'][:]
            input_list = split_MNIST_inputs(train_input, train_label)
            plot_einet_activations(
                einet=self.model.net,
                input_list=input_list,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name                
            )
        else:
            logging.info("No 'net' attribute on model. Skipping _plot_activations.")

    def _plot_gradients(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "grad_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data['train'][:]
            plot_einet_gradients(
                model=self.model,
                inputs=train_input,
                labels=train_label,
                save_root=save_dir,
                reshape_fn=None,
                reshape_exc_syn=True,
                save_in_dir=True,
                filename=self.file_name
            )
            logging.info(f"Saved gradient visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient plotting.")
    
    def _plot_profiles(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "profiles")
            os.makedirs(save_dir, exist_ok=True)
            shaper = Shaper(shape=(28, 28))
            plot_einet_profiles(
                model=self.model,
                train_data=self.data['train'],
                valid_data=self.data['valid'],
                n_tasks=1,
                logspace=True,
                reshape_fn=shaper.reshape,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name,
            )
            logging.info(f"Saved profile visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping profile plotting.")

    def _branch_info(self):
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping branch_info.")
            return
        net = self.model.net
        if not hasattr(net, 'forward_with_branch_outputs'):
            logging.info("No forward_with_branch_outputs found. Skipping branch_info analysis.")
            return
        logging.info("Computing branch mutual info ...")
        from dendritic_modeling.utils import analyze_branch_information
        branch_info = analyze_branch_information(
            net,
            self.data['train'],
            batch_size=getattr(self.config.train, 'batch_size', 64),
            device=self.device,
            n_bins=10
        )
        logging.info(f"Branch Info (MI) per layer: {branch_info}")
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)
        plot_depth_info(branch_info)
        plt.savefig(os.path.join(save_dir, "branch_info_depth_plot.png"))
        plt.close()
        logging.info(f"Saved branch_info_depth_plot.png in {save_dir}")

    def _ablation(self):
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping ablation.")
            return
        net = self.model.net
        if not hasattr(net, 'forward_with_branch_outputs'):
            logging.info("No forward_with_branch_outputs found. Skipping ablation.")
            return
        logging.info("Performing layer-wise ablation ...")
        from dendritic_modeling.utils import measure_performance
        baseline_acc = measure_performance(net, self.data['test'], device=self.device)
        logging.info(f"Ablation baseline acc on test = {baseline_acc}")
        original = {}
        for n, p in net.named_parameters():
            original[n] = p.detach().clone()
        layer_idxs = []
        perf_drops = []
        for i in range(len(net.branch_layers)):
            logging.info(f"Ablating direct inputs for layer {i} ...")
            if hasattr(net.branch_layers[i], 'branch_excitation'):
                net.branch_layers[i].branch_excitation.pre_w.data[:, :net.branch_layers[i].excitatory_input_dim] = 0.
            if hasattr(net.branch_layers[i], 'branch_inhibition'):
                net.branch_layers[i].branch_inhibition.pre_w.data[:, :net.branch_layers[i].inhibitory_input_dim] = 0.
            acc_after = measure_performance(net, self.data['test'], device=self.device)
            delta = baseline_acc - acc_after
            logging.info(f"Layer {i}: new acc={acc_after:.3f}, drop={delta:.3f}")
            layer_idxs.append(i)
            perf_drops.append(delta)
            for n, p in net.named_parameters():
                p.data.copy_(original[n])
        save_dir = os.path.join(self.save_path, "ablation_plots")
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(layer_idxs, perf_drops, marker='o', color='orange')
        ax.set_xlabel("Branch Layer Index")
        ax.set_ylabel("Accuracy Drop")
        ax.set_title("Ablation of Direct Input vs. Layer Index")
        fig.savefig(os.path.join(save_dir, "ablation_vs_layer_index.png"))
        plt.close(fig)
        logging.info(f"Saved ablation_vs_layer_index.png in {save_dir}")