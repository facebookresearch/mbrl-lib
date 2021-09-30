Models package
==============
This package provides implementations of common model architectures used in model-based RL,
including probabilistic and deterministic ensembles. All models in the library derive from
class :class:`mbrl.models.Model`. We provide a generic ensemble implementation,
:class:`mbrl.models.BasicEnsemble`, that can be used to produce epistemic uncertainty estimates
for any subclass of `Model`. For efficiency considerations, some specific model implementations
also provide their own ensemble implementations, without having to rely on BasicEnsemble.
One such model is :class:`mbrl.models.GaussianMLP`, which can be used as a single model or as
an ensemble. Additionally, it can be used as a deterministic model
trained with MSE loss, or a parameterized Gaussian with mean and log variance outputs, trained
with negative log-likelihood.

.. automodule:: mbrl.models
    :members:
    :show-inheritance:
