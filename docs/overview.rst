*******
SASNets
*******

**SASNets is a neural network implementation that classifies 1D SANS data as one of 71 SANS shape models, based on SASView and SASModels.** 

Features
--------

* **Cross-platform:** Fully written in Python 2.7, SASNets will run on any platform that has a working Python distribution.
* **Extensible:** SASNets uses the standard Keras and Tensorflow libraries, so it is easy to build upon and improve.
* **Multifunction:** While SASNets was originally developed for SANS data, it can be applied to SAXS and non-SAS datasets. 

Motivation
----------

Manual analysis of SAS data is difficult to do, especially with complex crystals and solutions. Convolutional Neural Networks have been shown to have great success on complex image recognition and retrieval tasks. We implement a novel combination of a convolutional neural network with 1D SAS data. The current iteration of the program was aimed at demonstrating the potential that CNNs could have on SAS fitting. Project extensions in the future will include expanding to 2D, tuning the network for even higher accuracy, and incorporating automatic bumps parameter fitting. 

Results
-------

SASNets has been able to achieve a prediction accuracy of between 40% and 50% on the 71 model set. It has also been able to achieve x percent accuracy on a simplified classification task.

Contact
-------
Chris Wang was the main author of the program, and can be reached at chrwang0 *at* icloud *dot* com.
The project advisors were Paul Kienzle and William Ratcliff, and can be reached at {paul.kienzle, william.ratcliff} *at* nist *dot* gov.

All code can be found online at https://github.com/scattering/sasnets. If you have any questions or contributions, file an issue on Github.

Acknowledgements
----------------
This project benefitted from the NSF-funded DANSE Project, DMR-0520547, SANS Subproject from the University of Tenessee Knoxville.
SASView and SASModels were developed by an international team as part of the DANSE project. Their work can be found at https://github.com/sasview. The custom fork of SASModels used in this program can be found at https://github.com/chrwang/sasmodels. 
