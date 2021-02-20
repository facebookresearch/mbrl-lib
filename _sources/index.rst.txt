Documentation for mbrl-lib
========================================
``mbrl-lib`` is library to facilitate research on Model-Based Reinforcement Learning.

Getting started
===============

Installation
------------

``mbrl-lib`` is a Python 3.7+ library. To install it, clone the repository,

.. code-block:: bash

    git clone https://github.com/fairinternal/mbrl-lib.git

then run

.. code-block:: bash

    cd mbrl-lib
    pip install -e .

If you also want the developer tools for contributing, run

.. code-block:: bash

    pip install -e ".[dev]"

Finally, make sure your Python environment has
`PyTorch (>= 1.6) <https://pytorch.org/>`_ installed with the appropriate CUDA configuration
for your system.


To test your installation, run

.. code-block:: bash

    python -m pytest tests

.. toctree::
   :maxdepth: 3
   :caption: Contents

   models.rst
   planning.rst
   math.rst
   util.rst
   replay_buffer.rst
   logging.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`