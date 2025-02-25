

<p align="center">
  <a href="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/deploy-docs.yml">
    <img src="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/deploy-docs.yml/badge.svg?branch=develop" alt="docs">
  </a>
  <a href="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/python-package.yml">
    <img src="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/python-package.yml/badge.svg" alt="tests">
  </a>
  <a href="https://codecov.io/gh/KempnerInstitute/dendritic-modeling" > 
 <img src="https://codecov.io/gh/KempnerInstitute/dendritic-modeling/graph/badge.svg?token=HUJRFX92DF"/> 
 </a>
</p>



This Python package introduces a novel approach to artificial neural networks by emulating dendritic computation through the encoding of dendrite morphology into the network archtecture. Inspired by the complexity and efficiency of biological neurons, our models aim to enhance traditional neural network architectures by incorporating dendrite-like structures.

## Key Features

Dendrite-Inspired Neural Networks: Implementations of neural networks that simulate the functionality and structure of dendrites, enabling more complex and nuanced information processing.
Customizable Morphologies: Users can define and experiment with various dendritic morphologies, allowing for the exploration of different computational properties and their impact on learning and performance.
Integration with Existing Frameworks: Seamlessly integrate with popular deep learning frameworks, making it easy to incorporate into existing projects and workflows.
Extensive Documentation and Examples: Comprehensive documentation and a collection of examples to help users get started quickly and explore the capabilities of the package.


## Setting Up a Mamba Virtual Environment
To ensure a consistent and isolated development environment, we recommend using Mamba for managing dependencies. Follow the steps below to set up a Mamba virtual environment for this project:

### Install Mamba
If you haven't installed Mamba yet, you can do so by using Conda:

```sh
    conda install mamba -n base -c conda-forge
```

### Create a Mamba Environment
Create a new Mamba environment with the desired Python version:

```sh
    mamba create -n <myvenv>
```
### Activate the Environment
Activate the newly created environment:
    
```sh    
    mamba activate <myvenv>
```

### Install the Package

```sh
    pip install .
```

If you want to install the package in development mode, use the following command:

```sh
    pip install -e .
```


## Generating Documentation Locally

To generate the documentation locally, you can use the following command:

```sh
    pip install -e '.[docs]'
    cd docs
    make html
    open _build/html/index.html
```

## Running Tests

To run the tests, you can use the following command:

```sh
    pip install -e '.[dev]'
    pytest
```

To run the tests with coverage, you can use the following command:

```sh
    pytest --cov=dendritic_modeling --cov-report=html
    open htmlcov/index.html
```

## Running on Kempner Cluster GPUs

The steps for running the package on the Kempner cluster GPU are similar to the previous instructions; however, you must ensure that compatible modules are loaded.

```sh
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
```

