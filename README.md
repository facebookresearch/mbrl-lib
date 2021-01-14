# MBRL-Lib

``mbrl-lib`` is a toolbox for facilitating development of 
Model-Based Reinforcement Learning algorithms. It provides easily interchangeable 
modeling and planning components, and a set of utility functions that allow writing
model-based RL algorithms with only a few lines of code. 

## Getting Started

### Installation

``mbrl-lib`` is a Python 3.7+ library. To install it, clone the repository,

    git clone https://github.com/fairinternal/mbrl-lib.git

then run

    cd mbrl-lib
    pip install -e .

If you also want the developer tools for contributing, run

    pip install -e ".[dev]"

Finally, make sure your Python environment has
[PyTorch (>= 1.6)](https://pytorch.org) installed with the appropriate 
CUDA configuration for your system.

    python -m pytest tests

### Basic example
As a starting point, check out our [tutorial notebook](notebooks/pets_example.ipynb) 
on how to write the PETS algorithm 
([Chua et al., NeurIPS 2018](https://arxiv.org/pdf/1805.12114.pdf)) 
using our toolbox, and running it on a continuous version of the cartpole 
environment.

## Documentation 
Please check out our documentation!

