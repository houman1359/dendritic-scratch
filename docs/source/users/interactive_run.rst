Running an interactive session
==============================

This section provides a step-by-step guide to running an interactive session with the Dendritic Modeling package. Interactive sessions are useful for exploring the package's features, testing different network configurations, and visualizing the results of your experiments.

To run a job with the `dendritic_modeling` package, we need the following components:

- A Python environment with the required dependencies installed
- A configuration file
- A run script 

In the following sections, we will walk you through the process of setting up an interactive session with the Dendritic Modeling package.

Step 1: Prepare your run environment
------------------------------------

The first step is setting up your compute environment. It can be your laptop or an an interactive session on the cluster. You will need to get the code and install the package in your conda environment. Please refer to the :ref:`installation_guide` for detailed instructions on setting up your environment.


Step 2: Create a configuration file

Each compute job requires a configuration file that specifies the network architecture, training parameters, and other experimental settings. The following shows an example of a configuration file:

.. code-block:: yaml

    model:
      task: classification         
      probabilistic: True
      network:
        type: "EINet"
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512]
          inhibitory_layer_sizes: [200]
          excitatory_branch_factors: [64]
          inhibitory_branch_factors: []
          reactivate: true
          somatic_synapses: true
          ee_synapses_per_branch_per_layer: [24]
          ei_synapses_per_branch_per_layer: [200]
          ie_synapses_per_branch_per_layer: [1]
          ii_synapses_per_branch_per_layer: []
          topk_init_method: xavier_normal
          use_shunting: true
          init_gain: 1.0
          init_inhib_offset: 1.3
          reactivation_initial_m: 2
          reactivation_initial_b: 0.5
          reactivation_type: param_tanh       
          train_reactivation_b_only: false
          gradient_strategy: block_conductance_dynamic  
          gradient_scale_factor: 2
          weight_decay_rate: 0.1
          enable_branch_outputs: true
          output_layer: bool = true
          output_dim: int = 10

    train:
      learning_strategy: "mle"           
      run_name: "mnist_einet"
      seed: 0
      train_valid_split: 0.8
      lr: 0.01
      epochs: 50
      batch_size: 128
      grad_clip_value: 5
      epochs_per_layer: 50            
      pretrain_epochs: 10             
      maintrain_epochs: 50            
      visualize_epochs: [1, 10, 50]   

    task:
      dataset: "mnist"
      parameters: {}

    wandb:
      entity: your_wandb_entity
      project: "dendritic_mnist_test"
      group: "debug"
      tags: [mnist_1]
      use_wandb: true

    visualization:
      enabled: true
      save_path: /path/to/results
      plots:
        - "weights"        
        - "activations"    
        - "gradients"      
        - "branch_info"    
        - "ablation"       


The configuration file defines three main sections:
  - **model**: Specifies the network architecture, including the type of network (e.g., EINet), layer sizes, branch factors, synaptic connections, and reactivation settings. Notice the new options for reactivation (e.g. using either a full parametric tanh or a version with only m trainable via "param_tanh_only_m").
  - **train**: Defines the training parameters, such as the learning strategy (e.g., "mle", "freeze_layers", "two_step", etc.), total number of epochs, batch size, and an additional list called "visualize_epochs" that determines at which epochs the visualization routines should run.
  - **wandb**: Contains settings for logging the experiment to Weights & Biases (W&B).
  - **visualization**: Contains settings for saving the generated plots and the list of plot types.

Please refer to the :ref:`concept_overview` for more details on the network architecture.

Step 3: Run the interactive session
-----------------------------------

Once you have set up your environment and created the configuration file, you can run the interactive session using the following command:

.. code-block:: bash

    $ python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml --output_dir results --experiment_name mnist_einet


The `output_dir` and `experiment_name` arguments specify the output directory and the name of the experiment, respectively. The `train_experiments.py` script will load the configuration file `config_exp.yaml` and start the training process. You can monitor the training progress and visualize the results using the W&B dashboard.


Step 4: Monitor outputs and visualizations
-------------------------------------------

During training, the log messages will indicate:
  - The completion of each epoch.
  - When an intermediate visualization is generated (including the epoch number and the type of plot, such as weights, activations, gradients, branch_info, or ablation).
  - The final performance metrics (train, validation, and test accuracy) will be logged.

Additionally, if W&B is enabled, you can monitor the training progress and visualizations on the W&B dashboard.

For further analysis, the generated plots (including loss curves, weight maps, activation distributions, gradient maps, branch information, and ablation results) are saved in the directory specified under the visualization section of the config file.




