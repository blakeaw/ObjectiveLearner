# ObjectiveLearner
#### Get more out of objective function evaluations.

![Python version badge](https://img.shields.io/badge/python->=3.6-blue.svg)
[![license](https://img.shields.io/github/license/blakeaw/ObjectiveLearner.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.1.0-orange.svg)
[![release](https://img.shields.io/github/release-pre/blakeaw/ObjectiveLearner.svg)](https://github.com/blakeaw/ObjectiveLearner/releases/tag/v0.1.0)

**ObjectiveLearner** is a python module that provides functionality to run machine learning (linear regression) and sensitivity analysis on the objective function versus parameter relationship using the thousands (to millions) of (sometimes expensive) objective function evaluations performed during model calibration with packages like [PyDREAM](https://github.com/LoLab-VU/PyDREAM), [simplePSO](https://github.com/LoLab-VU/ParticleSwarmOptimization), [Gleipnir](https://github.com/LoLab-VU/Gleipnir), and [GAlibrate](https://github.com/blakeaw/GAlibrate).

ObjectiveLearner provides easy to use objective function decorators which allow users to save data from the objective function evaluations performed during model calibration, and thereby provides them a way to utilize what would typically be lost data (i.e., not saved by the calibrator) and learn even more about the objective function and its relationship to model parameters, as well as learn more about the underlying model and assumptions the objective function represents.

------

# Install

| **! Warning** |
| :--- |
|  ObjectiveLearner is still under heavy development and may rapidly change. |

**ObjectiveLearner** installs as the `objlearner` package. It is compatible (i.e., tested) with Python >= 3.6.

Note that `objlearner` has the following core dependencies:
   * [NumPy](http://www.numpy.org/)
   * [pandas](https://pandas.pydata.org/)
   * [scikit-learn](https://scikit-learn.org/stable/)
   * [SALib](https://salib.readthedocs.io/en/latest/)

### pip install
You can install the latest release of the `objlearner` package using `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/blakeaw/ObjectiveLearner@v0.1.0#egg=objlearner
```
However, this will not automatically install the core dependencies. You will have to do that separately:
```
pip install numpy pandas scikit-learn SALib
```

### Recommended additional software

The following software is not required for the basic operation of ObjectiveLearner, but provides extra capabilities and features when installed.

#### PySB
[PySB](http://pysb.org/) is needed to run PySB models.


#### Jupyter
If you want to run the Jupyter IPython notebooks that come with ObjectiveLearner then you need to install [Jupyter](https://jupyter.org/).

------

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

------

# Documentation and Usage

### Quick Overview
Principally, **ObjectiveLearner** defines the **ObjectiveLearner** class,
```python
from objlearner import ObjectiveLearner
```
which defines an object that can be used to decorate a objective function:
```python
@ObjectiveLearner
def objective(theta):
    ...
    ...
    return objective_value
```
The ObjectiveLearner decorator saves the input parameter vectors and the corresponding objective values generated when the objective function is called during model calibrations and provides functions to post-analyze the values with linear regression (using tools from scikit-learn) and sensitivity analysis (using tools from SALib).

Additionally, **ObjectiveLearner** provides two other decorator classes:
    * CostCounter - simply keeps count of the number of calls to the objective function.
    * CostSaver - In addition to keeping count of the number of calls to the objective function, it stores the parameter vectors and corresponding objective function values and provides functions to write the data to a file.

### Jupyter Notebooks
Checkout the Jupyter Notebook:
 1. [Basics of ObjectiveLearner](./jupyter_notebooks/Basics_of_ObjectiveLearner.ipynb)

------

# Contact

To report problems or bugs, make comments, suggestions, or feature requests please open a [GitHub Issue](https://github.com/blakeaw/ObjectiveLearner/issues).

------

# Citing

If you use the **ObjectiveLearner** software in your research, please cite the GitHub
repository.

Also, please cite the following references as appropriate for software used with/via **ObjectiveLearner**:

#### Packages from the SciPy ecosystem
These include NumPy and pandas for which references can be obtained from:
https://www.scipy.org/citing.html

#### scikit-learn
  1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

#### SALib
  1. Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. Journal of Open Source Software, 2(9). doi:10.21105/joss.00097

#### PySB
  1. Lopez, C. F., Muhlich, J. L., Bachman, J. A. & Sorger, P. K. Programming biological models in Python using PySB. Mol Syst Biol 9, (2013). doi:[10.1038/msb.2013.1](dx.doi.org/10.1038/msb.2013.1)
