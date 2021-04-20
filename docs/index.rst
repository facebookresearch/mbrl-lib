Documentation for mbrl-lib
========================================
``mbrl-lib`` is library to facilitate research on Model-Based Reinforcement Learning.

Getting started
===============

Installation
------------

``mbrl-lib`` is a Python 3.7+ library. To install it, clone the repository,

.. code-block:: bash

    git clone https://github.com/facebookresearch/mbrl-lib.git

then run

.. code-block:: bash

    cd mbrl-lib
    pip install -e .

If you also want the developer tools for contributing, run

.. code-block:: bash

    pip install -e ".[dev]"

Finally, make sure your Python environment has
`PyTorch (>= 1.7) <https://pytorch.org/>`_ installed with the appropriate CUDA configuration
for your system.


To test your installation, run

.. code-block:: bash

    python -m pytest tests/core

Mujoco
------
Mujoco is a popular library for testing RL methods. Installing Mujoco is not
required to use most of the components and utilities in MBRL-Lib, but if you
have a working Mujoco installation (and license) and want to test MBRL-Lib
on it, you please install

.. code-block:: bash

    pip install -r requirements/mujoco.txt

and to test our mujoco-related utilities, run

.. code-block:: bash

    python -m pytest tests/mujoco

Basic Example
-------------
As a starting point, check out our
`tutorial notebook <https://github.com/facebookresearch/mbrl-lib/blob/master/notebooks/pets_example.ipynb>`_
on how to write the PETS algorithm
`(Chua et al., NeurIPS 2018) <https://arxiv.org/pdf/1805.12114.pdf>`_
using our toolbox, and running it on a continuous version of the cartpole
environment. Then, please take a look at our API documentation below.

.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   models.rst
   planning.rst
   math.rst
   util.rst
   replay_buffer.rst
   logging.rst
