# Changelog

## main (v0.2.0.dev3)
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

### Other changes
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