************
Installation
************

System Requirements
^^^^^^^^^^^^^^^^^^^
SASNets requires a system with a \*nix style command line. Any Unix/Unix-like distributions will satisfy this (Linux, macOS, RedHat, etc.). On Windows, you will either need the WSL (untested, but should work) or a terminal system such as MinGW or Cygwin.

You'll need either Python 2.7 or 3.6 installed. While SASNets can theoretically run on any hardware, we recommend at least 4 physical cores, at least a GTX 970 or similar card, and at least 16 GB of RAM. On systems with lower specifications, training will be extremely slow, and may crash on memory-constrained systems. Consider training on AWS or Google Cloud instances if you cannot obtain such a system physically. Any performance metrics cited in these docs were, unless otherwise stated, run on a GTX 1080 Ti, Intel i7-7700, and 32 GB of DDR4.

Install Procedure
^^^^^^^^^^^^^^^^^

Installing SASNets is relatively easy. First, clone the Github repository https://github.com/scattering/SASNets. We recommend installing SASNets in a virtualenv to keep its packages separate from others. Do a ::

  python setup.py install

which will install the SASNets and related dependencies. There are optional features (specified as extras in setup.py) which you can install to gain additional functionality. Currently, this consists of:

* **ruamel.yaml**: Used for better json parsing. Only relevant for sequential read from .json formatted files.
* **bumps**: Used for fitting data to theory models output by the neural network. Currently unimplemented.
* **Sphinx**: Used for building documentation locally. The docs are also online at https://sasnets.readthedocs.io.
* **seaborn**: Used in some matplotlib plots for more colourful output.

Note that setup.py installs the pip version of Tensorflow, which may or may not come with the features that you need. If you would like Nvidia GPU support, install tensorflow-gpu from pip and comment out tensorflow from setup.py. If you would like more advanced features such as OpenCL on CPUs and Intel GPUs or Google Cloud Platform support, compile from source. Note that Nvidia support for Tensorflow on macOS is deprecated.

It is highly recommended that you use the PostgreSQL version of SASNets, as this uses significantly less memory and is much faster. This requires you to have PostgreSQL installed. Consult your OS package manager documentation for how to do so.

If you have any problems, file an issue on Github or contact the developers directly.
