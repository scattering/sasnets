************
Installation
************

System Requirements
^^^^^^^^^^^^^^^^^^^
SASNets requires a system with a *nix style command line. Any Unix/Unix-like distributions will satisfy this (Linux, macOS, RedHat, etc.). On Windows, you will either need the WSL (untested, but should work) or a terminal system such as MinGW or Cygwin.

You'll need either Python 2.7 or 3.6 installed. While SASNets can theoretically run on any hardware, we recommend at least 4 physical cores, at least a GTX 970 or similar card, and at least 16 GB of RAM. On systems with lower specifications, training will be extremely slow, and may crash on memory-constrained systems. Consider training on AWS or Google Cloud instances if you cannot obtain such a system physically. Any performance metrics cited in these docs were, unless otherwise stated, run on a GTX 1080 Ti, Intel i7-7700, and 32 GB of DDR4. 

Install Procedure
^^^^^^^^^^^^^^^^^

Installing SASNets is relatively easy. First, clone the Github repository https://github.com/scattering/SASNets. We recommend installing SASNets in a virtualenv to keep its packages separate from others. Do a ::

  pip install -r requirements.txt

and optionally ::

  pip install -r optional-requirements.txt

which will install the necessary packages for SASNets.

Optionally, you can also run ::

  python setup.py install

if you would like SASNets installed as a dist-package.

It is highly recommended that you use the PostgreSQL version of SASNets, as this uses significantly less memory and is much faster. This requires you to have PostgreSQL installed. Consult your OS package manager documentation for how to do so.

SASNets is ready to use out of the box; no actual installation is necessary. See the individual function documentation for more information.

If you have any problems, file an issue on Github or contact the developers directly.
