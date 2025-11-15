.. image:: https://codecov.io/github/Roman-HLIS-Cosmology-PIT/furry-parakeet/graph/badge.svg?token=K33BI5YAKV

furry-parakeet: C kernels and wrappers for image combination
############################################################

This repository contains the linear algebra and interpolation kernels used by `PyIMCOM <https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom>`_.

Installation
============

You should be able to pip install by going to the ``furry-parakeet`` directory and running::

    pip install .

This uses the Numpy/C API. If you want to adjust the default settings for C compilers, you can edit ``setup.py`` (for example: openmp is enabled by default).

Legacy versions
===============

Previous versions of ``furry-parakeet`` installed each file as a python module, so you would import with ``import pyimcom_croutines``. You would now write ``from furry_parakeet import pyimcom_croutines``. You can make code that would be compatible with either using::

    try:
        from furry_parakeet import pyimcom_croutines
    except ImportError:
        import pyimcom_croutines

Helpful information
===================

There are `interface instructions <docs/interface_instructions.rst>`_ if you are interested in how ``furry-parakeet`` can interface with other tools in the Roman ecosystem.

You can also look at the `readthedocs page <https://furry-parakeet.readthedocs.io/en/latest/>`_ for the Python routines, and the `C routines page <docs/croutines.rst>`_ for the C routines that are wrapped into Python.

Links
=====

Some useful links include:

* `Hirata et al. (2024) <https://arxiv.org/abs/2303.08749>`_ describes the D5512 interpolator. The tables in Appendix A are available in machine-readable form `here <docs/Tables_A2A3.txt>`_.

* `Cao et al. (2025) <https://arxiv.org/abs/2410.05442>`_ describes the various linear algebra kernels.
