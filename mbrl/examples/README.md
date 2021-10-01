The `mbrl.examples` package can be used to train models using our example MBRL algorithm
implementations. We currently have examples for 
[PETS](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/algorithms/pets.py), 
[MBPO](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/algorithms/mbpo.py), 
and [PlaNet](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/algorithms/planet.py).

The examples can be run by typing

```bash
python -m mbrl.examples.main ${hydra_options}
```

where `${hydra_options}` is any set of 
[Hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli) overrides. To see the
available overrides, take a look at our 
[configuration files](https://github.com/facebookresearch/mbrl-lib/tree/main/mbrl/examples/conf).
The config files are generally structured in 4 groups:
* `algorithm`: includes options specific to each algorithm that typically don't 
  vary across experiments.
* `dynamics_model`: describes the dynamics model to use. 
* `overrides`: describes experiment specific configuration and hyperparameters. Typically, we 
  include one for each environment to be run, which we have populated with the best 
  hyper-parameters for each environment we have found so far.
* `action_optimizer`: describes possible optimizers to use for action selections. Some algorithms,
like MBPO, ignore this.
  
For example, to run MBPO on `gym`'s Hopper environment using the standard ensemble version of
[GaussianMLP](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/gaussian_mlp.py),
you can type

```bash
python -m mbrl.examples.main \
  algorithm=mbpo \
  overrides=mbpo_hopper \
  dynamics_model=gaussian_mlp_ensemble \
  algorithm.agent.batch_size=256 \
  overrides.validation_ratio=0.2 \
  dynamics_model.activation_fn_cfg._target_=torch.nn.ReLU
```
where we have re-written some defaults, just to show how `hydra` command line syntax
works. The number of possible options is extensive, and the best way to explore would be to 
look at the configuration files. 

Finally, keep in mind that not all models and algorithms are compatible, and the correct
combination needs to be specified manually in the command line. For example, running PlaNet 
requires passing both `algorithm=planet` and `dynamics_model=planet`, in addition to any 
other arguments you wish to change.

By default, all algorithms will save results in a csv file called `results.csv`,
inside a folder whose path looks like 
`./exp/mbpo/default/gym___HalfCheetah-v2/yyyy.mm.dd/hhmmss`; 
you can change the root directory (`./exp`) by passing 
`root_dir=path-to-your-dir`, and the experiment sub-folder (`default`) by
passing `experiment=your-name`. The logger will also save a file called 
`model_train.csv` with training information for the dynamics model.