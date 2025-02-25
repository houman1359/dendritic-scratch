Sweep Job Submission
====================

Sumbitting a sweep job is as easy of submitting a single job. All you need to do is to create a sweep configuration file and run `scripts/generate_sweep_configs.py` script, to setup the sweep job. In the following sections, we will walk you through the process of setting up a sweep job. 

Step 1: Prepare sweep configuration file
----------------------------------------

This file contains all necessary information to setup the sweep job. Here is an example of a sweep configuration file:

.. code-block:: yaml (NEED TO BE UPDATED WITH THE NEW CONFIG)

  slurm_config:
    output_dir:  # <--- Add your output directory path
    conda_env_path: # <--- Add your conda environment path
    modules_to_load: ["python/3.12.5-fasrc01", "cuda/12.4.1-fasrc01", "cudnn/8.9.2.26_cuda12-fasrc01"] # <--- Modify this based on your requirements
    max_concurrent_jobs: 8
    training_script: "scripts/train_mnist.py"
    config_filename_init: "test_config"

    account: "kempner_dev"
    nodes: 1
    ntasks_per_node: 1
    gpus_per_node: 1
    cpus_per_task: 10
    mem: 64GB
    time: "1:00:00"
    partition: "kempner"


  train_config:
    train:
      run_name: "dendritic_modeling_sweep"
      seed: 0
      epochs: 10
      batch_size: 32
    
    wandb:
      entity:  # <---- Add your entity
      project: dendritic_modeling_pr
      group: debug
      tags: [mnist_1]
  
    optimizer:
      name: "adam"
      learning_rate: 0.001
  
    scheduler:
      name: "none"
    
    model:
      input_dim: 784
      excitatory_layer_sizes: [10]
      inhibitory_layer_sizes: [20]
      excitatory_branch_factors: [24]
      inhibitory_branch_factors: []
      reactivate: true
      somatic_synapses: true
      ee_synapses_per_branch_per_layer: [24]
      ei_synapses_per_branch_per_layer: [100]
      ie_synapses_per_branch_per_layer: [1]
      ii_synapses_per_branch_per_layer: []
    
  sweep_config:
    optimizer:
      learning_rate: [0.001, 0.0001, 0.1]
    model:
      inhibitory_layer_sizes: [[10], [20], [30]]
      ee_synapses_per_branch_per_layer: [[24], [22]]


The sweep configuration file has three major sections: 

- `slurm_config`, 
- `train_config`, 
- and `sweep_config`. 

Slurm configuration
+++++++++++++++++++

In the slurm configuration section you need to provide all information to create the batch submission script. The parameters' name are self-explanatory. ``max_concurrent_jobs`` is the number of jobs that will run concurrently.

Train configuration
+++++++++++++++++++

The `train_config` section defines the training parameters, including the run name, seed, epochs, batch size, optimizer, scheduler, and model architecture.

Sweep configuration
+++++++++++++++++++

The `sweep_config` section contains the hyperparameters to sweep over, such as the learning rate, layer sizes, and synaptic connections. Note that you need to add extra bracket to the list of lists in the sweep configuration file. For example, `inhibitory_layer_sizes: [[10], [20], [30]]`, which means that the sweep will run with the values `[10]`, `[20]`, and `[30]`. For learning rate, you can sweep over a list of values `[0.001, 0.0001, 0.1]`, which means that the sweep will run with the values `0.001`, `0.0001`, and `0.1`.

Step 2: Generate sweep configureation files and slurm script
------------------------------------------------------------

After creating the sweep configuration file (let's say `sweep_params.yaml`), you can generate the sweep configuration files and slurm script using the following command:

.. code-block:: bash

    $ python scripts/generate_sweep_configs.py configs sweep_params.yaml


Running this command will generate a set of configuration files and a slurm script that will run the sweep job. The configuration files will be saved in the `configs` directory, and the slurm script will be saved in the current directory.

Step 3: Submit the sweep job

Once you have generated the sweep configuration files and the slurm script, you can submit the sweep job using the following command:

.. code-block:: bash

    $ sbatch run.sh


Navigate to the output directory (from `slurm_config`) to monitor the progress of the sweep jobs. You can also view the results in the Weights & Biases dashboard by logging in to your W&B account. Inside the output directory, there will be a directory named `[run_name]_[SLURM_ARRAY_JOB_ID]`, which contains subfolders `[SLURM_ARRAY_JOB_ID]_[SLURM_ARRAY_TASK_ID]`. All the results and logs will be saved in these subfolders.
