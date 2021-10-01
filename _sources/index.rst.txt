Documentation for mbrl-lib
========================================
``mbrl`` is library to facilitate research on Model-Based Reinforcement Learning.

Getting started
===============

Installation
------------

Standard Installation
^^^^^^^^^^^^^^^^^^^^^
``mbrl`` requires Python 3.7+ and `PyTorch (>= 1.7) <https://pytorch.org/>`_.

To install the latest stable version, run

.. code-block:: bash

    pip install mbrl

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^
If you are interested in modifying parts of the library, you can clone the repository
and set up a development environment, as follows

.. code-block:: bash

    git clone https://github.com/facebookresearch/mbrl-lib.git
    pip install -e ".[dev]"

And test it by running

.. code-block:: bash

    python -m pytest tests/core
    python -m pytest tests/algorithms


Basic Example
-------------
As a starting point, check out our
`tutorial notebook <https://github.com/facebookresearch/mbrl-lib/blob/main/notebooks/pets_example.ipynb>`_
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
   env.rst