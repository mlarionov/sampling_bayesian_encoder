# Sampling Bayesian Encoder
Experimental studies of my paper "Sampling Techniques in Bayesian Target Encoding" https://arxiv.org/abs/2006.01317

This repository contains only experiments. The actual implementation of Sampling Bayesian Encoder is located in a public 
fork of category_encoders package and can 
be accessed at this github location: https://github.com/mlarionov/categorical-encoding/tree/sampling . 

To get the latest version use:

`pip install -r requirements.txt`

## Experimental Studies

### Classification tasks

We use two synthetic data sets, one is generated using `make_classification()`, and another `make_hastie_10_2()` 
from `scikit-learn`. To run cross-validation run `classification_cv.py` and `hastie_cv.py`. 
It saves the cross-validation results in the `studies` folder. To run the studies, mentioned in the paper 
run  `classification_studies.py`. Of our interest is to see how hyperparameters affect model accuracy. 
Also we are interested in finding out how sampling techniques affect feature importance as reported 
by Random Forest models. The study results are also saved in the `studies` folder.

### Regression task

Rather than using synthetic data, we decided to use a data set from Kaggle's competition 
https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/ . To run it first download the data files
in the zip format to this folder, then run `regression-mercedes.py`. It also save the cross-validation results
to the `studies` folder. It also produces two submission files, both using Random Forest Regressor, 
but one used `LeaveOneOutEncoder` for the data preprocessing, the other uses `SamplingBayesianEncoder`.
 


