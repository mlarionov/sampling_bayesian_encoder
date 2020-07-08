#!/usr/bin/env python
# coding: utf-8

# In[5]:



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[6]:


all_data = pd.read_csv('train.csv.zip')
all_data.head(2)


# In[7]:


#Removing duplicate or constant columns as per https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554
columns_to_remove = ['X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347', 
                     'X382', 'X232', 'X279', 'X35', 'X37', 'X39', 'X302', 'X113', 'X134', 'X147', 'X222', 
                     'X102', 'X214', 'X239', 'X76', 'X324', 'X248', 'X253', 'X385', 'X172', 'X216', 'X213', 
                     'X84', 'X244', 'X122', 'X243', 'X320', 'X245', 'X94', 'X242', 'X199', 'X119', 'X227', 
                     'X146', 'X226', 'X326', 'X360', 'X262', 'X266', 'X247', 'X254', 'X364', 'X365', 'X296', 'X299',
                     'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']
new_columns = [col for col in all_data.columns if col not in columns_to_remove]
data1 = all_data[new_columns]


# In[8]:


data1.shape


# ## Baseline linear regression without categorical features

# In[9]:


X = data1.iloc[:,10:].values
y = data1.iloc[:,1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)


# In[11]:


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Train R^2: ', model.score(X_train, y_train))
print('Test R^2: ', r2_score(y_test, preds))
print('Test MSE: ', mean_squared_error(y_test, preds))


# ## Label encoding of all columns
# 

# In[12]:


from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
test_data = pd.read_csv('test.csv.zip')
combined = pd.concat([all_data, test_data], axis=0, sort=False)

cat_column_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

label_encoders = {}

for col in cat_column_names:
    label_encoder = LabelEncoder()
    label_encoder.fit(combined[col])
    label_encoders[col] = label_encoder


# ## Mean encoding of columns
# Now we will do a naive mean encoding of the categorical columns X0-X8

# In[13]:


data2 = data1.copy()


# In[14]:


import sys,os, pathlib
current = pathlib.Path(os.getcwd())
base = current.parent.parent
catenc = base.joinpath('categorical-encoding')
sys.path.append(str(catenc))


# In[15]:


from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder

cat_column_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

for col in cat_column_names:
    data2[col] = label_encoders[col].transform(data2[col])


X = data2.iloc[:,2:].values
y = data2.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)

mean_enc_columns = [data2.columns.get_loc(c) for c in data2.columns if c in cat_column_names]


m_encoder = LeaveOneOutEncoder(cols=mean_enc_columns)
m_encoder.fit(X_train, y_train)
X_train = m_encoder.transform(X_train)
X_test = m_encoder.transform(X_test)

scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha = 20) #Best alpha we could get via hyperparameter tuning
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Train R^2: ', model.score(X_train, y_train))
print('Test R^2: ', r2_score(y_test, preds))
print('Test MSE: ', mean_squared_error(y_test, preds))


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)

m_encoder = LeaveOneOutEncoder(cols=mean_enc_columns)
m_encoder.fit(X_train, y_train)
X_train = m_encoder.transform(X_train)
X_test = m_encoder.transform(X_test)

scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=55)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Train R^2: ', model.score(X_train, y_train))
print('Test R^2: ', r2_score(y_test, preds))
print('Test MSE: ', mean_squared_error(y_test, preds))


# The result is actually worse. Is it because we ignored the standard deviation?
# 
# We will try to achieve the best result by using random forest

# In[17]:


from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)


#m_encoder = LeaveOneOutEncoder(cols=mean_enc_columns)
m_encoder = TargetEncoder(cols=mean_enc_columns, smoothing=1E-2)
m_encoder.fit(X_train, y_train)
X_train = m_encoder.transform(X_train)
X_test = m_encoder.transform(X_test)


model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=2834, n_jobs=-1) 
model.fit(X_train, y_train)
preds = model.predict(X_test)

print('Train R^2: ', model.score(X_train, y_train))
print('Test R^2: ', r2_score(y_test, preds))
print('Test MSE: ', mean_squared_error(y_test, preds))


# In[18]:


#generate submission
kaggle1 = test_data[new_columns[2:]].copy()

for col in cat_column_names:
    kaggle1[col] = label_encoders[col].transform(kaggle1[col])
    
X_kaggle = kaggle1.values


# In[19]:


X_kaggle_tran = m_encoder.transform(X_kaggle)
preds_kaggle = model.predict(X_kaggle_tran)
preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
preds_kaggle_df.head(2)
preds_kaggle_df.to_csv('te_submission.csv', index=False)


# This is much better than Ridge regression, but without limiting the max depth it overfits tremendously

# In[20]:



from category_encoders.posterior_imputation import PosteriorImputationEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)


m_encoder = PosteriorImputationEncoder(cols=mean_enc_columns, n_draws=25, prior_samples_ratio=0.01, random_state=2834)
m_encoder.fit(X_train, y_train)
X_train = m_encoder.transform(X_train)
X_test = m_encoder.transform(X_test)
y_train = m_encoder.expand_y(y_train)

#print(X_train.isnull().mean())

model = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=2834, n_jobs=-1) 
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds = m_encoder.average_y(preds)

print('Train R^2: ', model.score(X_train, y_train))
print('Test R^2: ', r2_score(y_test, preds))
print('Test MSE: ', mean_squared_error(y_test, preds))


# I do not see much difference. We need another example where we can prove better effectiveness of this algorithm.

# In[21]:


X_kaggle_tran = m_encoder.transform(X_kaggle)
preds_kaggle = model.predict(X_kaggle_tran)
preds_kaggle = m_encoder.average_y(preds_kaggle)
preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
preds_kaggle_df.head(2)
preds_kaggle_df.to_csv('pm_submission.csv', index=False)


# ## Cross validation
# 
# We will do cross-validation to ensure validity of comparison between the two approaches

# In[22]:


mean_enc_column_names = [c for c in data2.columns if c in cat_column_names]


# ### Leave one out encoding

# In[31]:



from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from category_encoders.leave_one_out import LeaveOneOutEncoder
import optuna
from optuna.distributions import *


loo = LeaveOneOutEncoder(cols=mean_enc_column_names,  random_state=2834)
rf = RandomForestRegressor(n_estimators=400, random_state=2834, n_jobs=-1) 
pipe = Pipeline(steps=[('loo',loo), ('rf',rf)])


param_distribution = {
    'loo__sigma': LogUniformDistribution(1E-5, 1E-1),
    'rf__max_depth': IntUniformDistribution(2,40),
    'rf__max_features' : IntUniformDistribution(1,X_test.shape[1]),
    'rf__min_samples_leaf': IntUniformDistribution(1,15)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=data1.columns[2:])
X_test = pd.DataFrame(X_test, columns=data1.columns[2:])

search = optuna.integration.OptunaSearchCV(pipe, param_distribution, 
                            cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout=20*60, scoring='r2')
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
test_predict = search.best_estimator_.predict(X_test)
print('Test R^2: ', r2_score(y_test, test_predict))


# In[32]:


preds_kaggle = search.best_estimator_.predict(pd.DataFrame(X_kaggle, columns=data1.columns[2:]))
preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
preds_kaggle_df.head(2)
preds_kaggle_df.to_csv('te517_submission.csv', index=False)


# In[33]:


from category_encoders.pte_utils import EncoderWrapperR
from sklearn.model_selection import cross_val_score

pte = PosteriorImputationEncoder(cols=mean_enc_column_names, random_state=2834)
model = RandomForestRegressor(n_estimators=400, random_state=2834, n_jobs=-1) 
wrapper_model = EncoderWrapperR(pte, model)

param_distribution = {
    'encoder__prior_samples_ratio': LogUniformDistribution(1E-9, 1E-1),
    'encoder__n_draws': IntUniformDistribution(1,40),
    'encoder__include_precision': CategoricalDistribution([False, True]),
    'regressor__max_depth': IntUniformDistribution(2,40),
    'regressor__max_features' : IntUniformDistribution(1,X_test.shape[1]),
    'regressor__min_samples_leaf': IntUniformDistribution(1,15)
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
X_train = pd.DataFrame(X_train, columns=data1.columns[2:])
X_test = pd.DataFrame(X_test, columns=data1.columns[2:])

search = optuna.integration.OptunaSearchCV(wrapper_model, param_distribution, 
                cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout=2*60*60, scoring='r2')

search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
test_predict = search.best_estimator_.predict(X_test)
print('Test R^2: ', r2_score(y_test, test_predict))


# In[34]:


preds_kaggle = search.best_estimator_.predict(pd.DataFrame(X_kaggle, columns=data1.columns[2:]))
preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
preds_kaggle_df.head(2)
preds_kaggle_df.to_csv('sa517_submission.csv', index=False)


# In[ ]:




