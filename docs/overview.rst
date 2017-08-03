*******
SASNets
*******

**SASNets is a neural network implementation that classifies 1D SANS data as one of 71 SANS shape models, based on SASView and SASModels.**

Features
--------

* **Cross-platform:** Fully written in Python, SASNets will run on any platform that has a working Python distribution.
* **Extensible:** SASNets uses the standard Keras and Tensorflow libraries, so it is easy to build upon and improve.
* **Multifunction:** While SASNets was originally developed for SANS data, it can be applied to SAXS and non-SAS datasets.

Goal
----

Streamline the process of SANS and SAXS data analysis through the use of a convolutional neural network (CNN).

Motivation
----------

Manual analysis of SAS data is difficult to do, especially with complex crystals and solutions. Convolutional Neural Networks have been shown to have great success on complex image recognition and retrieval tasks. We implement a novel combination of a convolutional neural network with 1D SAS data. The current iteration of the program was aimed at demonstrating the potential that CNNs could have on SAS fitting. Project extensions in the future will include expanding to 2D, tuning the network for even higher accuracy, and incorporating automatic bumps parameter fitting.

Results
-------

SASNets has been able to achieve a prediction accuracy of between 40% and 50% on the 71 model set. We also demonstrate that the network is able to distinguish superclusters or types of models, for example cylinders or spheres.

More Information
----------------

For a more in-depth exploration of this project, please see :download:`this poster <source/SASNets.pdf>` or :download:`this presentation <source/classifying-sans-data.pdf>`.

A full size version of the dendrogram can be found :download:`here <source/dendrogram.pdf`.  

License
-------

The poster, presentation, and this documentation are released under the CC-BY-SA 4.0 license. The codebase of SASNets is released under the BSD 3-clause license.

Contact
-------
Chris Wang was the main author of the program, and can be reached at com *period* icloud *asperand* chrwang0, backwards.
The project advisors were Paul Kienzle and William Ratcliff of NCNR.

All code can be found online at https://github.com/scattering/sasnets. If you have any questions or contributions, file an issue on Github, or send me(Chris) an email.

Acknowledgements
----------------
This project benefited from the NSF-funded DANSE Project, DMR-0520547, SANS Subproject from the University of Tennessee Knoxville.

This project benefited from funding and material support from the NIST Centre for Neutron Research and the NSF-funded CHRNS.

SASView and SASModels were developed by an international team as part of the NSF-funded DANSE project. Their work can be found at https://github.com/sasview. The custom fork of SASModels used in this program can be found at https://github.com/chrwang/sasmodels.

Finally, thanks to all of the people who helped read, test, and critique the project.
