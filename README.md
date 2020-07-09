# Sampling Bayesian Encoder
Experimental studies of my paper "Sampling Techniques in Bayesian Target Encoding" https://arxiv.org/abs/2006.01317

This repository contains only experiments. The actual implementation of Sampling Bayesian Encoder is located in a public 
fork of category_encoders package and can 
be accessed at this github location: https://github.com/mlarionov/categorical-encoding/tree/sampling . 

To get the latest version use:

`pip install --upgrade 'git+https://github.com/mlarionov/categorical-encoding@sampling'`

## Experimental Studies

### Classification tasks

We use two synthetic data sets, one is generated using `make_classification()`, and another `make_hastie_10_2()` from `scikit-learn`.
To run cross-validation run `classification_cv.py`. It saves the model in the pickle format. To run the studies, 
mentioned in the paper run  `classification_studies.py`. Of our interest is to see how hyperparameters affect
model accuracy. Also we are interested in finding out how sampling techniques affect feature importance as
reported by Random Forest models.


