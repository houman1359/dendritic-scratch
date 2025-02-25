import os
import argparse
from pathlib import Path
from copy import deepcopy
from itertools import product

from omegaconf import OmegaConf, DictConfig

def read_config(directory = "configs", 
                param_file_name = "sweep_params.yaml"):
    """
    Reads the configuration file from the specified directory.

    Parameters
    ----------

    directory : str
        The directory containing the configuration file. Default is "configs".
    
    param_file_name : str
        The name of the configuration file. Default is "sweep_params.yaml".
    """
    
    config_path = os.path.join(directory, param_file_name)
    
    try:
        assert os.path.exists(config_path), f"Config file not found at {config_path}"
        params = OmegaConf.load(config_path)
        return params
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def custom_yaml_dump(data, indent=0):
        """
        Custom YAML dump function to print a dictionary as YAML,
        with lists inline if they are encountered.

        Parameters
        ----------

        data : dict
            The dictionary to print as YAML.

        indent : int
            The current(initial) indentation level. Default is 0.

        Returns
        -------

        yaml_str: str
            The YAML string representation of the dictionary.
        """

        yaml_str = ""
        indent_str = "  " * indent  # Two spaces for each indentation level
        
        for key, value in data.items():
            if isinstance(value, dict) or isinstance(value, DictConfig):
                # Recursively process dictionaries with increased indentation
                yaml_str += f"{indent_str}{key}:\n"
                yaml_str += custom_yaml_dump(value, indent + 1)
            elif isinstance(value, list):
                # Write lists inline with brackets
                yaml_str += f"{indent_str}{key}: {value}\n"
            else:
                # Write scalar values directly
                yaml_str += f"{indent_str}{key}: {value}\n"
        
        return yaml_str

def validate_params(params):
    """
    Validates the presence of required configurations in the parameters.

    Parameters:
    params (dict): A dictionary containing the configuration parameters.

    Raises:
    ValueError: If any of the required configurations ('slurm_config', 
    'train_config') are missing from the parameters.

    Example:
    try:
        validate_params(params)
    except ValueError as e:
        print(e)
    """
    required_configs = ['slurm_config', 'train_config']
    for config in required_configs:
        if config not in params:
            raise ValueError(f"Missing required config: {config}")
        
def flatten(config, parent_key='', sep='.'):
    """
    Flattens a nested dictionary or OmegaConf DictConfig into a single level 
    dictionary with keys as the path to each value.

    Parameters
    ----------
 
    config : dict or DictConfig
        The nested dictionary or OmegaConf DictConfig to flatten.

    present_key : str
        The base key string for the nested keys. Default is an empty string.

    sep : str
        The separator between keys. Default is '.'.
    """

    if config is None:
        return {}
    
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, DictConfig):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)
        

def generate_config_files(base_config, 
                          sweep_params, 
                          output_dir="config_files"):
    """
    Generates configuration files for each combination of sweep parameters.

    Parameters
    ----------
    base_config : dict
        The base configuration dictionary.

    sweep_params : dict
        The dictionary containing the sweep parameters.

    output_dir : str
        The directory to save the configuration files. 
        Default is "config_files".

    """
    os.makedirs(output_dir, exist_ok=True)

    sweep_keys = list(sweep_params.keys())
    sweep_values = [params for params in sweep_params.values()]
  

    # If nested lists, create all possible combinations
    all_combinations = list(product(*sweep_values))

    configs = []
    sweeps = []

    for combination in all_combinations:
        new_config = deepcopy(base_config)
        param_details = {}
        for key, value in zip(sweep_keys, combination):
            keys = key.split(".")
            sub_config = new_config
            for sub_key in keys[:-1]:
                sub_config = sub_config[sub_key]
            sub_config[keys[-1]] = value
            param_details[key] = value

        containerized_config = OmegaConf.to_container(new_config, resolve=False)
        configs.append(containerized_config)
        sweeps.append(param_details)

        
    print(f"Generated {len(configs)} configuration files.")
    return configs, sweeps


def write_config_files(configs,
                       sweeps,  
                       output_dir="configs", 
                       base_name="config"):
    """
    Writes the configuration files to disk.

    Parameters
    ----------

    configs : list
        A list of configuration dictionaries.

    sweeps : list
        A list of dictionaries containing the sweep parameters.

    output_dir : str
        The directory to save the configuration files. 
        Default is "configs".
    
    base_name : str
        The base name for the configuration files. Default is "config".
    """
    
    assert len(configs) == len(sweeps), "Number of configs and sweeps do not match."

    os.makedirs(output_dir, exist_ok=True)

    sweep_lines = []

    for i, config in enumerate(configs):
        config_name = f"{base_name}_{i}.yaml"
        config_path = os.path.join(output_dir, config_name)
        config_yaml = custom_yaml_dump(config)
        with open(config_path, "w") as f:
            f.write(config_yaml)
        sweep_lines.append(f"{config_name}: {sweeps[i]} \n")
        print(f"Configuration saved to {config_path}")

    sweep_path = os.path.join(output_dir, f"{base_name}_sweeps.txt")
    sweep_lines = "".join(sweep_lines)
    with open(sweep_path, "w") as f:
        f.writelines(sweep_lines)


def generate_run_sh(num_config, 
                    config_dir, 
                    config_filename_init, 
                    slurm_config):
    """
    Generates a SLURM batch script for running jobs based on the given 
    configurations.

    Parameters
    ----------
    
    num_config: int
        The number of generated configurations.

    config_dir: str
        The directory where the configuration files are located.

    config_file_name_init: str
        The initial name of the configuration files.

    slurm_config: dict
        A dictionary containing SLURM configuration options.
    """

    array_job_flag = False
    if num_config > 1:
        array_job_flag = True

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_config['run_name']}",
        f"#SBATCH --account={slurm_config['account']}"
    ]

    if array_job_flag:
        lines.extend([
            f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/output_%A_%a.out",
            f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/error_%A_%a.out",
        ])
    else:
        lines.extend([
            f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/output_%j.out",
            f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/error_%j.out",
        ])

    lines.extend([f"#SBATCH --nodes={slurm_config['nodes']}",        
                  f"#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}",
                  f"#SBATCH --gpus-per-node={slurm_config['gpus_per_node']}",     
                  f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}",
                  f"#SBATCH --time={slurm_config['time']}",
                  f"#SBATCH --mem={slurm_config['mem']}",
                  f"#SBATCH --partition={slurm_config['partition']}"])

    # Submit an array job or a single job
    if array_job_flag:
        if num_config > slurm_config['max_concurrent_jobs']:
            lines.append(f"#SBATCH --array=0-{num_config-1}%{slurm_config['max_concurrent_jobs']}")
        else:
            lines.append(f"#SBATCH --array=0-{num_config-1}")

    lines.append("")
    
    lines.append("# ================================================================ ")
    lines.append("# This file has been generated by generate_sweep_configs.py script ")
    lines.append("#                            ***                                   ")
    lines.append("#       Manual changes to this file may be overwritten.            ")
    lines.append("# ================================================================ ")


    # Add module loading lines
    lines.append("module purge")
    for module in slurm_config['modules_to_load']:
        lines.append(f"module load {module}")
    
    lines.append("")
    lines.append(f"conda activate {slurm_config['conda_env_path']}")
    lines.append("")
    lines.append("# Path to the config file directory")
    lines.append(f"CONFIG_DIR={config_dir}")
    lines.append("")
    lines.append(f"# Path to output directory")
    lines.append(f"OUTPUT_DIR={slurm_config['output_dir']}")
    lines.append("")
    lines.append(f"CONFIG_FILE_NAME_INIT={config_filename_init}")
    lines.append(f"SC_RUN_NAME={slurm_config['run_name']}")
    lines.append("")
    lines.append("#Training script: ")
    lines.append(f"TRAINING_SCRIPT={slurm_config['training_script']}")
    lines.append("")
    lines.append("#Python path: ")
    lines.append(f"PYTHON={slurm_config['conda_env_path']}/bin/python")
    lines.append("")
    lines.append("# Get the specific config file based on the array job ID")
    if array_job_flag:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}')
    else:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_0.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_JOB_ID}')
    lines.append("")
    lines.append(f"# Path to the folder to save outputs")
    if array_job_flag:
        lines.append(f"OUTPUT_FOLDER_PATH={slurm_config['output_dir']}/${{SC_RUN_NAME}}_${{SLURM_ARRAY_JOB_ID}}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}")
    else:
        lines.append(f"OUTPUT_FOLDER_PATH={slurm_config['output_dir']}/${{SC_RUN_NAME}}_${{SLURM_JOB_ID}}")
    lines.append("")
    lines.append('echo "Starting running a training job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")
    lines.append("  ${PYTHON} -u  ${TRAINING_SCRIPT} ${CONFIG} \\")
    lines.append("  --output_dir=${OUTPUT_FOLDER_PATH} \\")
    lines.append("  --experiment_name=${RUN_NAME} \\")
    lines.append("  ${@}")
    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done with running a training job at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')
    
    with open('run.sh', 'w') as file:
        for line in lines:
            file.write(line + "\n")
    
    print("run.sh file has been generated.")


def parse_args():
    """
    """

    parser = argparse.ArgumentParser(description="Generate configuration files for a sweep.")
    parser.add_argument("config_dir", 
                        default="configs", 
                        type=str, 
                        help="The directory containing the configuration file.")
    
    parser.add_argument("param_filename",
                        default="sweep_params.yaml",
                        type=str,
                        help="The name of the sweep parameters file.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    config_dir = args.config_dir
    param_filename = args.param_filename
    
    # Read the config file
    params = read_config(config_dir, param_filename)

    # Validate the parameters
    validate_params(params)

    params.slurm_config.run_name = params.train_config.train.run_name

    # Generate configuration files for each combination of sweep parameters
    base_config = params['train_config']
    sweep_params = params['sweep_config']
    slurm_config = params['slurm_config']
    config_filename_init = slurm_config['config_filename_init']

    flatten_sweep_params = flatten(sweep_params)
    configs, sweeps = generate_config_files(base_config, 
                                            flatten_sweep_params, 
                                            output_dir=config_dir)
    

    # Write the configuration files to disk
    write_config_files(configs, 
                       sweeps,
                       output_dir=config_dir, 
                       base_name=config_filename_init )
    

    # Generate a SLURM batch script for running jobs
    # Get config file name init    
    generate_run_sh(len(configs), 
                    config_dir=config_dir, 
                    config_filename_init=config_filename_init, 
                    slurm_config=slurm_config)

    