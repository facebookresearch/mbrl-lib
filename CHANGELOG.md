# Changelog

## v0.1.4
- Added MPPI optimizer.
- `control_env.py` now works with CEM and MPPI.
- Changed algorithm configuration so that action optimizer is passed as another 
  config file.

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