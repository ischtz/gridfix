# GridFix


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.998554.svg)](https://doi.org/10.5281/zenodo.998554)


GridFix is a Python 3 toolbox to facilitate preprocessing of scene fixation data for region-based analysis using Generalized Linear Mixed Models (GLMM).

[Our recently published manuscript](https://www.frontiersin.org/articles/10.3389/fnhum.2017.00491) [1] describes how this approach can be used to evaluate models of visual saliency above and beyond content-independent biases. Please also see [3] for a previous description of the approach and [our ECVP 2016 poster](http://doi.org/10.5281/zenodo.571067) [2] for an overview about the structure and workflow of the GridFix toolbox.

[![Example Image](https://ischtz.github.io/gridfix/_images/example_grid.png)]


## Features
- Define image parcellations (region masks) based on a grid or bounding box coordinates (RegionSet)
- Apply these parcellations to collections of images or saliency maps (ImageSet)
- Calculate per-region fixation status (fixated/not fixated), count, and duration-based dependent measures
- Define features to assign a value to each region X image (e.g., mean saliency of each region)
- Explicitly model central viewer bias using different approaches (e.g. Euclidean distance, Gaussian)
- Output the resulting feature vectors for GLMM-based analysis [1,2]
- Create initial R source code to facilitate GLMM analysis using lme4


## Installation

GridFix requires Python 3. The following packages need to be installed: 
- numpy
- scipy
- matplotlib
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


## Documentation

See [GridFix documentation](https://ischtz.github.io/gridfix/). 
Tutorial Jupyter Notebooks and example data are [available as a separate download](https://github.com/ischtz/gridfix-tutorial/releases).


## References
[1] Nuthmann, A., Einhäuser, W., & Schütz, I. (2017). How well can saliency models predict fixation selection in scenes beyond central bias? A new approach to model evaluation using generalized linear mixed models. Frontiers in Human Neuroscience. http://doi.org/10.3389/fnhum.2017.00491

[2] Schütz, I., Einhäuser, W., & Nuthmann, A. (2016). GridFix: A Python toolbox to facilitate fixation analysis and evaluation of saliency algorithms using Generalized linear mixed models (GLMM). Poster presented at the European Conference on Visual Perception (ECVP), Barcelona. http://doi.org/10.5281/zenodo.571067

[3] Nuthmann, A., & Einhäuser, W. (2015). A new approach to modeling the influence of image features on fixation selection in scenes. Annals of the New York Academy of Sciences, 1339(1), 82-96. http://dx.doi.org/10.1111/nyas.12705

