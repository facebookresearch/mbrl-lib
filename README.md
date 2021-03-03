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

If you are interested in contributing, please install the developer tools as well

    pip install -e ".[dev]"

Finally, make sure your Python environment has
[PyTorch (>= 1.7)](https://pytorch.org) installed with the appropriate 
CUDA configuration for your system.

    python -m pytest tests

### Basic example
As a starting point, check out our [tutorial notebook](notebooks/pets_example.ipynb) 
on how to write the PETS algorithm 
([Chua et al., NeurIPS 2018](https://arxiv.org/pdf/1805.12114.pdf)) 
using our toolbox, and running it on a continuous version of the cartpole 
environment.

## Provided baselines
MBRL-Lib provides implementations of popular MBRL algorithms 
as examples of how to use this library. You can find them in the 
[mbrl/algorithms](mbrl/algorithms) folder. Currently, we have implemented
[PETS](mbrl/algorithms/pets.py) and [MBPO](mbrl/algorithms/mbpo.py), and
we plan to keep increasing this list in the near future.

The implementations rely on [Hydra](https://github.com/facebookresearch/hydra) 
to handle configuration. You can see the configuration files in 
[this](conf) folder. The [overrides](conf/overrides) subfolder contains
environment specific configurations for each algorithm, with the best 
hyperparameter values we have found so far for each. You can run training
by passing the desired override option via command line.

For example, to run MBPO on the gym version of HalfCheetah, you should call
```python
python main.py algorithm=mbpo overrides=mbpo_halfcheetah 
```
By default, this will save results in a folder that looks like 
`./exp/mbpo/default/gym___HalfCheetah-v2/yyyy.mm.dd/hhmm`; 
you can change the root directory (`./exp`) by passing 
`root_dir=path-to-your-dir`, and the experiment sub-folder (`default`) by
passing `experiment=your-name`. You can also change other configuration options 
not in the overrides, such as the type of dynamics model by passing 
`dynamics_model=basic_ensemble`, or the number of models in the ensemble as 
`dynamics_model.model.ensemble_size=some-number` (when the dynamics model is
`gaussian_mlp_ensemble`). To learn more about the
all the available options, take a look at the provided [configuration files](conf). 


## Documentation 
Please check out our **[documentation](https://luisenp.github.io/mbrl-lib/)**!

## License
`mbrl-lib` is released under the MIT license. See [LICENSE](LICENSE) for 
additional details about it. See also our 
[Terms of Use](https://opensource.facebook.com/legal/terms) and 
[Privacy Policy](https://opensource.facebook.com/legal/privacy).
