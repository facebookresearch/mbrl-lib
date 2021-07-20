# Changelog

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