Environment package
===================
This package contains some useful wrappers, as well as termination and rewards
functions that can be passed to :class:`mbrl.models.ModelEnv` to use for simulation,
if needed. One useful class is :class:`mbrl.env.mujoco_envs.MujocoGymPixelWrapper`,
which facilitates the use of pixel-based observations in `mujoco-gym` environments.

Additionally, for completion, we provide a number of custom environments that have been
used in the original papers of the algorithms we have implemented. The environments
are provided to match their original versions, with the only changes related to the
Mujoco version (we only support 2.0). **Whenever possible, we suggest to use the
standard implementations of these environments in libraries like** ``mujoco-gym``,
``dmcontrol``, **rather than using the environments in this folder**. These are mostly
provided for completeness, and to facilitate debugging of the algorithms provided.
The current custom environments are:

* ``cartpole_continuous``: a basic continuous version of ``gym``'s cartpole environment.
  The only change is that the force applied is multiplied by an action in the range (-1, 1).
* ``ant_truncated_obs` and ``humanoid_truncated_obs``: these are the versions of ``Ant-v2``
  and ``Humanoid-v2`` used in the original MBPO paper, which removes some dimensions
  from the full observation.
* ``pets_halfcheetah``, ``pets_cartpole``, ``pets_reacher``, and ``pets_pusher`` are the
  environments used in the original PETS paper, which include some observation
  pre-processing functions.

.. automodule:: mbrl.env
    :members:
