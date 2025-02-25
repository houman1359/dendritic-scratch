.. _installation_guide:  

Installation
============  


To install the package:

- Step 1: Clone the repository

  + For developers: 

  .. code-block:: bash
    
        $ git clone git@github.com:KempnerInstitute/dendritic_modeling.git
    
  + For users:

  .. code-block:: bash
    
        $ git clone https://github.com/KempnerInstitute/dendritic_modeling.git
    

- Step 2: Create a conda environment

The simplified source code is not in a package format, so you need to install all packages in your conda environment. Here are the steps:

    - **Step a**: Load required libraries
      - You need to load the required libraries using the following command:
        
        .. code-block:: bash

            module load python/3.10.12-fasrc01
            module load cuda/12.2.0-fasrc01
            module load cudnn/8.9.2.26_cuda12-fasrc01
        
    - **Step b**: Create a conda environment
      - You can create a conda environment using the following command:
        
        .. code-block:: bash

            conda create --prefix [path_to_your_env] python=3.10
        
      - *Note*: Sometimes you might get the following installation error for `ffcv` package
        
        .. code-block:: bash

            RuntimeError: Could not find required package: opencv4.
        

        To fix that you can use the following command to create the conda environment:
        
        .. code-block:: bash

            conda create --prefix [path_to_your_env] python=3.10 pkg-config opencv -c conda-forge
        
    - **Step c**: Activate the conda environment
      - You can activate the conda environment using the following command:
        
        .. code-block:: bash

            conda activate [path_to_your_env]
        
    
    - **Step d**: Install the required packages
      
      - You can install the required packages using the following command:
        
        .. code-block:: bash
        
            pip install -r requirements.txt
    
- Step 3: Install the package

   + To install the package, you need to run the following command:

   .. code-block:: shell
    
        $ pip install -e .


Build the documentation 
======================= 

- Step 1: Install the required packages

.. code-block:: shell

    pip install -e '.[docs]'


- Step 2: Build the documentation

.. code-block:: shell

    cd docs
    make html


- Step 3: Open the documentation in your browser

.. code-block:: shell

    open _build/html/index.html
