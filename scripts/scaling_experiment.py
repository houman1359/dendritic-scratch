#!/usr/bin/env python3
"""
scaling_experiment.py
---------------------
- Runs scaling experiments for BOTH EINet and MLP on the same dimension list.
- For each dimension 'dim', we set 'excitatory_branch_factors = [2, dim]'.
- We parse param_count and final loss from the experiment logs.
- Then plot them in log–log space on a single figure:
    x = log(param_count)
    y = log(loss)
  with separate curves for EINet vs MLP.
  
Usage Example:
  python scripts/scaling_experiment.py \
    --base_config .vscode/config_exp.yaml \
    --output_dir /n/holylabs/LABS/kempner_dev/Users/hsafaai/results/results_scaling \
    --dim_list 2 4 \
    --experiment_name_prefix scaling_exp
"""

import os
import sys
import json
import argparse
import yaml
from copy import deepcopy
import matplotlib.pyplot as plt

# Import your train_experiments pipeline
from dendritic_modeling.train_experiments import main as train_main

def load_config_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config_yaml(cfg_dict, path):
    with open(path, 'w') as f:
        yaml.dump(cfg_dict, f, sort_keys=False)

def parse_experiment_logs(exp_dir):
    """
    Parse param_count and final_loss from logs or JSON in exp_dir.
    This is a placeholder; adapt to how your code actually logs them.
    For example:
      - "best_model_info.json" or "final_results.json" with fields 
        { "param_count": 123456, "final_loss": 0.321 }
    We'll look for 'final_results.json'.
    """
    param_count = None
    final_loss = None

    final_path = os.path.join(exp_dir, "final_results.json")
    if os.path.exists(final_path):
        with open(final_path, 'r') as f:
            data = json.load(f)
        # Suppose it logs "param_count" and "loss" or "final_loss"
        param_count = data.get("param_count", None)
        # you might store "test_nll" or "valid_loss" or something else:
        final_loss = data.get("final_loss", None)
        if final_loss is None:
            # fallback to e.g. "test_nll"
            final_loss = data.get("test_nll", None)
    return param_count, final_loss

def run_single_experiment(config_base, output_dir, experiment_name):
    """
    Saves a temp config, calls train_main, parses logs for param_count, final_loss.
    Returns (param_count, final_loss) or (None, None) if fail.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    save_config_yaml(config_base, config_path)

    try:
        train_main(config_path=config_path, output_dir=output_dir, experiment_name=experiment_name)
    except Exception as e:
        print(f"[ERROR] training failed: {e}")
        return None, None
    
    # parse logs
    pcount, loss = parse_experiment_logs(output_dir)
    return pcount, loss

def run_scaling_experiment(base_config_path, output_base_dir, dim_values,
                           experiment_name_prefix="scaling_exp"):
    """
    For each dim in dim_values, do 2 runs:
      1) EINet
      2) MLP
    Modify 'excitatory_branch_factors = [2, dim]' in config.
    Return a dictionary with keys 'EINet' and 'MLP', each a list of
    (param_count, final_loss, dim).
    """
    os.makedirs(output_base_dir, exist_ok=True)

    base_cfg = load_config_yaml(base_config_path)

    # We'll store results in e.g.:
    # results["EINet"] = [ { "dim":..., "param_count":..., "loss":...}, ...]
    # results["MLP"]   = [ { "dim":..., "param_count":..., "loss":...}, ...]
    results = {"EINet": [], "MLP": []}

    for dim in dim_values:
        # For each net type, we create a fresh config copy
        for net_type in ["EINet", "MLP"]:   # <-- FIXED HERE
            cfg = deepcopy(base_cfg)

            # set the network type
            cfg["model"]["network"]["type"] = net_type

            # set excitatory_branch_factors = [2, dim]
            # Make sure the config structure is correct
            cfg["model"]["network"]["parameters"]["excitatory_branch_factors"] = [2, dim]

            # (Optional) if we must also fix inhibitory_branch_factors to avoid index issues:
            cfg["model"]["network"]["parameters"]["inhibitory_branch_factors"] = []

            # Possibly also ensure excit/inhib layer sizes if needed
            # e.g. if your base_config had good default sizes, you might leave them alone.

            # We'll build a unique experiment dir for net_type & dim
            exp_name = f"{experiment_name_prefix}_{net_type}_dim{dim}"
            exp_dir  = os.path.join(output_base_dir, exp_name)

            pcount, loss = run_single_experiment(cfg, exp_dir, exp_name)
            print(loss)
            if (pcount is not None) and (loss is not None):
                results[net_type].append({"dim": dim, "param_count": pcount, "loss": loss})
            else:
                print(f"[WARNING] Could not get param_count/loss for {net_type} dim={dim}")

    return results

def plot_loglog(results, output_dir, experiment_name_prefix):
    """
    We'll make a single plot with 2 curves (EINet vs MLP).
    x-axis = log(param_count)
    y-axis = log(loss)
    """
    plt.figure(figsize=(6,5))

    for net_type, color, marker in [("EINet","blue","o"), ("MLP","red","s")]:
        data_list = results.get(net_type, [])
        # sort by param_count for a nice curve
        data_list = [d for d in data_list if (d["param_count"] and d["loss"])]
        data_list.sort(key=lambda x: x["param_count"])

        print(data_list)
        
        if len(data_list)==0:
            continue

        x_param = [d["param_count"] for d in data_list]
        y_loss  = [d["loss"] for d in data_list]

        # log scale
        plt.xscale("log")
        plt.yscale("log")

        plt.plot(x_param, y_loss, marker=marker, color=color, label=net_type)

    plt.xlabel("Param Count (log scale)")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Scaling: {experiment_name_prefix} (EINet vs MLP)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig = os.path.join(output_dir, f"{experiment_name_prefix}_loglog.png")
    plt.savefig(savefig)
    plt.show()
    print(f"[INFO] Figure saved at {savefig}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True, help="Base config path")
    parser.add_argument("--output_dir", type=str, default="results_scaling")
    parser.add_argument("--dim_list", type=int, nargs="+", default=[64,128,256,512])
    parser.add_argument("--experiment_name_prefix", type=str, default="scaling_exp")
    args = parser.parse_args()

    # 1) run all experiments
    results = run_scaling_experiment(
        base_config_path=args.base_config,
        output_base_dir=args.output_dir,
        dim_values=args.dim_list,
        experiment_name_prefix=args.experiment_name_prefix
    )

    # 2) Save overall results to JSON
    out_json = os.path.join(args.output_dir, f"{args.experiment_name_prefix}_allresults.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # 3) Plot log–log
    plot_loglog(results, args.output_dir, args.experiment_name_prefix)

    print("[DONE] All scaling runs complete and plotted.")

if __name__ == "__main__":
    main()