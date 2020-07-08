#!/usr/bin/env python
# coding: utf-8

# # Simulation studies using Hastie's data
# 
# The goal of this simulation study is to see how the performance of the Bayesian-encoded model varies for different values of the hyperparameters. 
# 

# In[1]:



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


import sys,os, pathlib
current = pathlib.Path(os.getcwd())
base = current.parent.parent
catenc = base.joinpath('categorical-encoding')
sys.path.append(str(catenc))


# # Binary classification problem
# 
# For Binary classifier we will work with the example 10.2 of T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning Ed. 2", Springer, 2009.

# In[3]:


from sklearn.datasets import make_hastie_10_2
X_h, y_h = make_hastie_10_2(random_state=2834)
X_h = X_h.astype('float16')
y_h[y_h==-1]=0


# In[4]:


#Now convert the last column to the categorical
from sklearn.preprocessing import KBinsDiscretizer
disczr1 = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
cat_column1 = disczr1.fit_transform(X_h[:,-1].reshape(-1, 1)) * 193 % 20 #We want to break the monotonicity
disczr2 = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
cat_column2 = disczr2.fit_transform(X_h[:,-2].reshape(-1, 1)) * 173 % 20 #We want to break the monotonicity


# In[5]:


predictors = pd.DataFrame(X_h[:, 0:-2], columns=[f'col_{i}' for i in range(8)])
predictors['cat1'] = cat_column1
predictors['cat2'] = cat_column2
#predictors['cat1_orig'] = cat_column1
#predictors['cat2_orig'] = cat_column2
predictors.head(3)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
model = RandomForestClassifier(n_estimators=400, max_depth=40, random_state=2834, n_jobs=-1) 
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:,1]

print('Train accuracy: ', accuracy_score(y_train, model.predict(X_train)))
print('Test accuracy: ', accuracy_score(y_test, preds.round()))
print('AUC: ', roc_auc_score(y_test, preds).round(4))


# Hyperparameter tuning: optimizing for AUC
# estimators: 400
# max depth:
# * 15 | 0.9334 
# * 17 | 0.9384
# * 19 | 0.9398
# * 21 | 0.9415
# * 25 | 0.9449
# * 30 | 0.947
# * 10 | 0.9476

# ## Cross-validation
# 
# We really should use cross-validation to avoid overfitting

# ### Cross-validation of the target encoding model
# 
# First we will train a model using target encoding

# In[25]:



from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from category_encoders.leave_one_out import LeaveOneOutEncoder
import optuna
from optuna.distributions import *

loo = LeaveOneOutEncoder(cols=['cat1', 'cat2'], sigma=0.05, random_state=2834)
rf = RandomForestClassifier(n_estimators=400, max_depth=30, max_features=1, min_samples_leaf=1,
                            random_state=2834, n_jobs=-1) 
pipe = Pipeline(steps=[('loo',loo), ('rf',rf)])

param_distribution = {
    'loo__sigma': LogUniformDistribution(1E-5, 1E-1),
    'rf__max_depth': IntUniformDistribution(5,40),
    'rf__max_features' : IntUniformDistribution(1,10),
    'rf__min_samples_leaf': IntUniformDistribution(1,10)
}

X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=predictors.columns)
X_test = pd.DataFrame(X_test, columns=predictors.columns)
search = optuna.integration.OptunaSearchCV(pipe, param_distribution, 
                    cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout= 20*60, scoring='accuracy')
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
test_predict = search.best_estimator_.predict(X_test)
print('Test accuracy: ', accuracy_score(y_test, test_predict))


# ### Cross-validation of the probabilistic encoder
# 
# First we create a class that makes it easier for us to run sklearn cross validation

# In[10]:


from category_encoders.pte_utils import EncoderWrapper
from category_encoders.posterior_imputation_bc import PosteriorImputationEncoderBC


# In[11]:


best_params = {'encoder__prior_samples_ratio': 2.649927356403324e-07, 'encoder__n_draws': 39, 'encoder__leave_one_out': False, 
               'classifier__max_depth': 38, 'classifier__max_features': 1, 'classifier__min_samples_leaf': 10} 


# In[27]:



from sklearn.model_selection import cross_val_score

if best_params is None:

    pte = PosteriorImputationEncoderBC(cols=['cat1', 'cat2'], random_state=2834)
    model = RandomForestClassifier(n_estimators=400, random_state=2834, n_jobs=-1) 
    wrapper_model = EncoderWrapper(pte, model)


    param_distribution = {
        'encoder__prior_samples_ratio': LogUniformDistribution(1E-10, 1E-1),
        'encoder__n_draws': IntUniformDistribution(1,40),
        'encoder__leave_one_out': CategoricalDistribution([False, True]),
        'classifier__max_depth': IntUniformDistribution(5,40),
        'classifier__max_features' : IntUniformDistribution(1,10),
        'classifier__min_samples_leaf': IntUniformDistribution(1,10)
    }


    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)

    search = optuna.integration.OptunaSearchCV(wrapper_model, param_distribution, 
                            cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout=90*60, scoring='accuracy')

    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    test_predict = search.best_estimator_.predict(X_test)
    print('Test accuracy: ', accuracy_score(y_test, test_predict))
    
    best_params = search.best_params_


# ## Study how hyperparameters influence the model

# First we check whether leave one out even matters

# In[12]:


from sklearn.model_selection import GridSearchCV
pte = PosteriorImputationEncoderBC(cols=['cat1', 'cat2'], 
                                   n_draws=best_params['encoder__n_draws'], 
                                   random_state=2834, 
                                   prior_samples_ratio=best_params['encoder__prior_samples_ratio'],
                                   leave_one_out=best_params['encoder__leave_one_out'])
model = RandomForestClassifier(n_estimators=400, 
                               max_depth=best_params['classifier__max_depth'], 
                               max_features=best_params['classifier__max_features'], 
                               min_samples_leaf=best_params['classifier__min_samples_leaf'], 
                               random_state=2834, n_jobs=-1) 
wrapper_model = EncoderWrapper(pte, model)


X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=predictors.columns)
X_test = pd.DataFrame(X_test, columns=predictors.columns)


search1 = GridSearchCV(estimator=wrapper_model, param_grid={'encoder__leave_one_out' : [False, True]})

search1.fit(X_train, y_train)
print(search1.cv_results_['mean_test_score'])
print(search1.cv_results_['std_test_score'])


# In[13]:


print(search1.cv_results_['mean_test_score'])
print(search1.cv_results_['std_test_score'])


# Seems like both results are equal within the statistical error
# 
# Now we will check number of samples

# In[14]:



X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=predictors.columns)
X_test = pd.DataFrame(X_test, columns=predictors.columns)

pte = PosteriorImputationEncoderBC(cols=['cat1', 'cat2'], 
                                   n_draws=best_params['encoder__n_draws'], 
                                   random_state=2834, 
                                   prior_samples_ratio=best_params['encoder__prior_samples_ratio'],
                                   leave_one_out=best_params['encoder__leave_one_out'])
model = RandomForestClassifier(n_estimators=400, 
                               max_depth=best_params['classifier__max_depth'], 
                               max_features=best_params['classifier__max_features'], 
                               min_samples_leaf=best_params['classifier__min_samples_leaf'], 
                               random_state=2834, n_jobs=-1) 
wrapper_model = EncoderWrapper(pte, model)

param_range = range(2, 20)
search2 = GridSearchCV(estimator=wrapper_model, param_grid={'encoder__n_draws' : param_range})

search2.fit(X_train, y_train)
results = search2.cv_results_['mean_test_score']    
    

plt.figure(1, figsize=(15, 10))
plt.plot(list(param_range), results, 'b-o')
plt.fill_between(list(param_range), results - search2.cv_results_['std_test_score'],
                 results + search2.cv_results_['std_test_score'], color='lightgrey')
plt.show();


# Seems like the algorithm favors small samples and not large samples, but above 5 the results are statistically the same
# 
# Now we will check how the prior distribution affects the accuracy
# 

# In[15]:



X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=predictors.columns)
X_test = pd.DataFrame(X_test, columns=predictors.columns)

pte = PosteriorImputationEncoderBC(cols=['cat1', 'cat2'], 
                                   n_draws=best_params['encoder__n_draws'], 
                                   random_state=2834, 
                                   prior_samples_ratio=best_params['encoder__prior_samples_ratio'],
                                   leave_one_out=best_params['encoder__leave_one_out'])
model = RandomForestClassifier(n_estimators=400, 
                               max_depth=best_params['classifier__max_depth'], 
                               max_features=best_params['classifier__max_features'], 
                               min_samples_leaf=best_params['classifier__min_samples_leaf'], 
                               random_state=2834, n_jobs=-1) 
wrapper_model = EncoderWrapper(pte, model)

param_range =  range(-10, -1)
search3 = GridSearchCV(estimator=wrapper_model, param_grid={'encoder__prior_samples_ratio' : [10**i for i in param_range]})

search3.fit(X_train, y_train)
    
results = search3.cv_results_['mean_test_score']    
    

plt.figure(1, figsize=(15, 10))
plt.plot(list(param_range), results, 'b-o')
plt.fill_between(list(param_range), results - search3.cv_results_['std_test_score'],
                 results + search3.cv_results_['std_test_score'], color='lightgrey')
plt.show();


# Seems like the algorithm favors the prior below $10^{-3}$ and within that range the results are not statistically different

# In[17]:


import pickle
with open('hastie_search1.pickle', 'wb') as pickle_file:
    pickle.dump(search1.cv_results_, pickle_file)
with open('hastie_search2.pickle', 'wb') as pickle_file:
    pickle.dump(search2.cv_results_, pickle_file)
with open('hastie_search3.pickle', 'wb') as pickle_file:
    pickle.dump(search3.cv_results_, pickle_file)


# In[ ]:




