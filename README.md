# Predictive Hyper Parameter Optimisation

[![Build Status](https://travis-ci.org/DobromirM/Predictive-Hyperparameter-Optimisation.svg?branch=master)](https://travis-ci.org/DobromirM/Predictive-Hyperparameter-Optimisation)

Implementation of a newly developed Predictive Hyperparameter Optimisation algorithm.

https://pypi.org/project/predictiveopt/

## Installation

`pip install predictiveopt`


## Building instructions

1) Building the package: `python setup.py sdist bdist_wheel`
2) Uploading to PyPi: `python -m twine upload  dist/*`
3) Uploading new versions: `twine upload --skip-existing dist/*`