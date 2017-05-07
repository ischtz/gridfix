GridFix is a Python 3 toolbox to facilitate preprocessing of scene fixation data for the grid-based analysis approach first described in Nuthmann & Einhäuser (2015)[1]. 

A manuscript describing the toolbox in detail is currently undergoing review. In the mean time, please feel free to take a look at [our ECVP 2016 poster](http://doi.org/10.5281/zenodo.571067)[2] for an overview of GridFix functionality. The GridFix documentation with examples is available here: https://ischtz.github.io/gridfix/.

## Features
- Define image parcellations (region masks) based on a grid or bounding box coordinates (RegionSet)
- Apply these parcellations to collections of images or saliency maps (ImageSet)
- Define features to assign a value to each region X image (e.g., mean saliency of each region)
- Explicitly model central viewer bias using different approaches (e.g. euclidean distance, Gaussian)
- Output the resulting feature vectors for GLMM-based analysis [1]
- Create initial R source code for GLMM analysis using lme4


## Installation

GridFix requires Python 3. The following packages need to be installed: 
- numpy
- scipy
- matplotlin
- pandas
- PIL or Pillow (image library)

If you need to first install a Python environment, we recommend the Anaconda distribution which includes all of the listed prerequisites by default, is available for all major platforms (Windows, Mac, Linux) and can be downloaded at https://www.continuum.io/downloads. 

To install GridFix, clone this repository using git or download a ZIP file using the "Clone or download" button, then run the following:

```
python3 setup.py install
```
or, in case you want to only install for your user or receive an error regarding permissions:

```
python3 setup.py install --user
```

GridFix was installed correctly if you can execute 

```
from gridfix import *
```

in an interactive Python 3 session without errors. 


## References
[1] Nuthmann, A., & Einhäuser, W. (2015). A new approach to modeling the influence of image features on fixation selection in scenes. Annals of the New York Academy of Sciences, 1339(1), 82-96. http://dx.doi.org/10.1111/nyas.12705

[2] Schütz, I., Einhäuser, W., & Nuthmann, A. (2016). GridFix: A Python toolbox to facilitate fixation analysis and evaluation of saliency algorithms using Generalized linear mixed models (GLMM). Poster presented at the European Conference on Visual Perception (ECVP), Barcelona. http://doi.org/10.5281/zenodo.571067
