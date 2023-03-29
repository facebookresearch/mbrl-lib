# Changelog

## main (v0.2.0)
### Breaking changes
- Migrated from [gym](https://github.com/openai/gym) to [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/)
- `gym==0.26.3` is still required for the dm_control and pybullet-gym environments
- `Transition` and `TransitionBatch` now support the `terminated` and `truncated` booleans
  instead of the single `done` boolean previously used by gym
- Migrated calls to `env.reset()` which now returns a tuple of `obs, info` instead of just `obs`
- Migrated calls to `env.step()` which now returns a `observation, reward, terminated, truncated, info`
- Migrated to Gymnasium render API, environments are instantiated with `render_mode=None` by default
- DMC and PyBullet envs use the original gym wrappers to turn them into gym environments, then are wrapper by gymnasium.envs.GymV20Environment
- All Mujoco envs use the DeepMind Mujoco [bindings](https://github.com/deepmind/mujoco), [mujoco-py](https://github.com/openai/mujoco-py) is deprecated as a dependency
- Custom Mujoco envs e.g. `AntTruncatedObsEnv` inherit from gymnasium.envs.mujoco_env.MujocoEnv, and access data through `self.data` instead of `self.sim.data`
- Mujoco environment versions have been updated to `v4` from`v2` e.g. `Hopper-v4`
- [Fixed](https://github.com/facebookresearch/mbrl-lib/blob/ac58d46f585cc90c064b8c989e7ddf64f9e330ce/mbrl/algorithms/planet.py#L147) PlaNet to save model to a directory instead of a file name
- Added `follow-imports=skip` to `mypy` CI test to allow for gymnasium/gym wrapper compatibility
- Bumped `black` to version `0.23.1` in CI

## v0.2.0.dev4
### Main new features
- Added [PlaNet](http://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) implementation.
- Added support for [PyBullet](https://pybullet.org/wordpress/) environments. 
- Changed SAC library used by MBPO 
  (now based on [Pranjan Tadon's](https://github.com/pranz24/pytorch-soft-actor-critic)).
  
### Breaking changes
- `Model.reset()` and `Model.sample()` signature has changed. They no longer receive
`TransitionBatch` objects, and they both return a dictionary of strings to tensors 
  representing a model state that should be passed to `sample()` to simulate 
  transitions. This dictionary can contain things like previous actions, predicted
  observation, latent states, beliefs, and any other such quantity that the model
  need to maintain to simulate trajectories when using `ModelEnv`. 
- `Ensemble` class and sub-classes are assumed to operate on 1-D models.
- Checkpointing format used by `save()` and `load()` in classes 
  `GaussianMLP` and `OneDTransitionRewardModel` changed, making old checkpoints 
  incompatible with the new version.
- `use_silu` argument to `GaussianMLP` has been replaced by `activation_fn_cfg`, which
is an `omegaconf.DictConfig` specifying the class to use for the activation functions, 
  thus giving more flexibility.
- Removed unnecessary nesting inside `dynamics_model` Hydra configuration.
- SAC agents prior to v0.2.0 cannot be loaded anymore.

### Other changes
- Added `add_batch()` method to `mbrl.util.ReplayBuffer`.
- Added functions to `mbrl.util.models` to easily create convolutional encoder/decoders
  with a desired configuration.
- `mbrl.util.common.rollout_agent_trajectories` now allows rolling out a pixel-based
environment using a policy trained on its corresponding non-pixel environment version.
- `ModelTrainer` can be given `eps` for `Adam` optimizer. It now also includes a
  progress bar using `tqdm` (can be turned off).
- CEM optimizer can now be toggled between using clipped normal distribution or
truncated normal distribution.
- `mbrl.util.mujoco.make_env` can now create an environment specified via an `omegaconf`
configuration and `hydra.utils.instantiate`, which takes precedence over the old
  mechanism if both are present.
- Fixed bug that assigned wrong termination functino to `humanoid_truncated_obs` env. 

## v0.1.4
- Added MPPI optimizer.
- Added iCEM optimizer.  
- `control_env.py` now works with CEM, iCEM and MPPI.
- Changed algorithm configuration so that action optimizer is passed as another 
  config file.
- Added option to quantize pixel obs of gym mujoco and dm control env wrappers.
- Added a sequence iterator, `SequenceTransitionSampler`, that always returns a 
  fixed number of random batches.

## v0.1.3
- Methods `loss`, `eval_score` and `update` of `Model` class now return a 
  tuple of loss/score and metadata. Currently, supports the old version as well,
  but this will be deprecated in v0.2.0.
- `ModelTrainer` now accepts a callback that will be called after every batch 
  both during training and evaluation.
- `Normalizer` in `util.math` can now operate using double precision. Utilities 
  now allow specifying if replay buffer and normalizer should use double or float 
  via Hydra config.

## v0.1.2
- Multiple bug fixes
- Added a training browser to compare results of multiple runs
- Deprecated `ReplayBuffer.get_iterators()` and replaced with `mbrl.util.common.get_basic_iterators()`
- Added an iterator that returns batches of sequences of transitions of a given length

## v0.1.1
- Multiple bug fixes
- Added `third_party` folder for `pytorch_sac` and `dmc2gym` 
- Library now available in `pypi`
- Moved example configurations to package `mbrl.examples`, which can now be
run as `python -m mbrl.examples.main`, after `pip` installation
  
## v0.1.0

Initial release
