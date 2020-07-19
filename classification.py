import pandas as pd
import pickle
from sklearn.datasets import make_classification,make_hastie_10_2
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.sampling_bayesian import SamplingBayesianEncoder, EncoderWrapper, TaskType
import optuna
from optuna.distributions import *
from sklearn.model_selection import GridSearchCV

# Some constants
rs_split = 8379
rs_enc = 1179
rs_rf = 5991
n_samples = 10000


def create_data():
    # In this experiment we will use standard classification synthetic data
    X_h, y_h = make_classification(n_samples=n_samples, n_features=10, n_informative=5, n_redundant=0,
                                   class_sep=0.01, random_state=2834)
    # Now convert the last column to the categorical
    disczr1 = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    cat_column1 = disczr1.fit_transform(X_h[:, -1].reshape(-1, 1)) * 193 % 20  # We want to break the monotonicity
    disczr2 = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
    cat_column2 = disczr2.fit_transform(X_h[:, -2].reshape(-1, 1)) * 173 % 20  # We want to break the monotonicity
    predictors = pd.DataFrame(X_h[:, 0:-2], columns=[f'col_{i}' for i in range(8)])
    predictors['cat1'] = cat_column1
    predictors['cat2'] = cat_column2
    # Uncomment the two lines if you want to keep the original columns
    # predictors['cat1_orig'] = cat_column1
    # predictors['cat2_orig'] = cat_column2
    return predictors, y_h


def create_data_hastie():
    X_h, y_h = make_hastie_10_2(random_state=2834)
    X_h = X_h.astype('float16')
    y_h[y_h == -1] = 0
    disczr1 = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    cat_column1 = disczr1.fit_transform(X_h[:, -1].reshape(-1, 1)) * 193 % 20  # We want to break the monotonicity
    disczr2 = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
    cat_column2 = disczr2.fit_transform(X_h[:, -2].reshape(-1, 1)) * 173 % 20  # We want to break the monotonicity
    predictors = pd.DataFrame(X_h[:, 0:-2], columns=[f'col_{i}' for i in range(8)])
    predictors['cat1'] = cat_column1
    predictors['cat2'] = cat_column2
    # Uncomment the two lines if you want to keep the original columns
    # predictors['cat1_orig'] = cat_column1
    # predictors['cat2_orig'] = cat_column2
    return predictors, y_h


def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=rs_split)
    model = RandomForestClassifier(n_estimators=100, max_depth=2, max_features=3, min_samples_leaf=1,
                                   random_state=rs_rf, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    print('Train accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test accuracy: ', accuracy_score(y_test, predictions.round()))
    print('AUC: ', roc_auc_score(y_test, predictions))
    return model


def cv_leave_one_out_encoder(predictors, y_h):
    loo = LeaveOneOutEncoder(cols=['cat1', 'cat2'], sigma=0.05, random_state=2834)
    rf = RandomForestClassifier(n_estimators=400, random_state=2834, n_jobs=-1)
    pipe = Pipeline(steps=[('loo', loo), ('rf', rf)])
    param_distribution = {
        'loo__sigma': LogUniformDistribution(1E-5, 1E-1),
        'rf__max_depth': IntUniformDistribution(5, 40),
        'rf__max_features': IntUniformDistribution(1, 10),
        'rf__min_samples_leaf': IntUniformDistribution(1, 3)
    }
    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)
    loo_search = optuna.integration.OptunaSearchCV(pipe, param_distribution,
                                                   cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout=5 * 60,
                                                   scoring='accuracy')
    loo_search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % loo_search.best_score_)
    print(loo_search.best_params_)
    test_predict = loo_search.best_estimator_.predict(X_test)
    print('Test accuracy: ', accuracy_score(y_test, test_predict))
    return loo_search


def cv_sampling(predictors, y_h):
    pte = SamplingBayesianEncoder(cols=['cat1', 'cat2'], n_draws=5, random_state=2834, prior_samples_ratio=0,
                                  task=TaskType.BINARY_CLASSIFICATION)
    model = RandomForestClassifier(n_estimators=400, max_depth=30, max_features=1,
                                   random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)
    param_distribution = {
        'encoder__prior_samples_ratio': LogUniformDistribution(1E-9, 1E-1),
        'encoder__n_draws': IntUniformDistribution(1, 40),
        'encoder__mapper': CategoricalDistribution(['mean', 'weight_of_evidence']),
        'estimator__max_depth': IntUniformDistribution(5, 40),
        'estimator__max_features': IntUniformDistribution(1, 10),
        'estimator__min_samples_leaf': IntUniformDistribution(1, 10)
    }
    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)
    search = optuna.integration.OptunaSearchCV(wrapper_model, param_distribution,
                                               cv=5, n_jobs=-1, random_state=514, n_trials=None, timeout=90 * 60,
                                               scoring='accuracy')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    test_predict = search.best_estimator_.predict(X_test)
    print('Test accuracy: ', accuracy_score(y_test, test_predict))
    return search


def study_mapper(predictors, y_h, search, filename):
    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)
    pte = SamplingBayesianEncoder(cols=['cat1', 'cat2'],
                                  n_draws=search.best_params_['encoder__n_draws'],
                                  random_state=2834,
                                  prior_samples_ratio=search.best_params_['encoder__prior_samples_ratio'],
                                  mapper=search.best_params_['encoder__mapper']
                                  )
    model = RandomForestClassifier(n_estimators=400,
                                   max_depth=search.best_params_['classifier__max_depth'],
                                   max_features=search.best_params_['classifier__max_features'],
                                   min_samples_leaf=search.best_params_['classifier__min_samples_leaf'],
                                   random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)
    param_range = ['mean', 'weight_of_evidence']
    grid_search = GridSearchCV(estimator=wrapper_model, param_grid={'encoder__mapper': param_range})
    grid_search.fit(X_train, y_train)
    with open(filename, 'wb') as pickle_file:
        pickle.dump(grid_search.cv_results_, pickle_file)


def study_n_draws(predictors, y_h, search, filename):
    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)
    pte = SamplingBayesianEncoder(cols=['cat1', 'cat2'],
                                  n_draws=search.best_params_['encoder__n_draws'],
                                  random_state=2834,
                                  prior_samples_ratio=search.best_params_['encoder__prior_samples_ratio'],
                                  mapper=search.best_params_['encoder__mapper']
                                  )
    model = RandomForestClassifier(n_estimators=400,
                                   max_depth=search.best_params_['classifier__max_depth'],
                                   max_features=search.best_params_['classifier__max_features'],
                                   min_samples_leaf=search.best_params_['classifier__min_samples_leaf'],
                                   random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)
    param_range = range(2, 20)
    grid_search = GridSearchCV(estimator=wrapper_model, param_grid={'encoder__n_draws': param_range})
    grid_search.fit(X_train, y_train)
    with open(filename, 'wb') as pickle_file:
        pickle.dump(grid_search.cv_results_, pickle_file)


def study_prior_samples_ratio(predictors, y_h, search, filename):
    X_train, X_test, y_train, y_test = train_test_split(predictors.values, y_h, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=predictors.columns)
    X_test = pd.DataFrame(X_test, columns=predictors.columns)
    pte = SamplingBayesianEncoder(cols=['cat1', 'cat2'],
                                  n_draws=search.best_params_['encoder__n_draws'],
                                  random_state=2834,
                                  prior_samples_ratio=search.best_params_['encoder__prior_samples_ratio'],
                                  mapper=search.best_params_['encoder__mapper']
                                  )
    model = RandomForestClassifier(n_estimators=400,
                                   max_depth=search.best_params_['classifier__max_depth'],
                                   max_features=search.best_params_['classifier__max_features'],
                                   min_samples_leaf=search.best_params_['classifier__min_samples_leaf'],
                                   random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)
    param_range = range(-10, -1)
    grid_search = GridSearchCV(estimator=wrapper_model,
                               param_grid={'encoder__prior_samples_ratio': [10 ** i for i in param_range]})
    grid_search.fit(X_train, y_train)

    with open(filename, 'wb') as pickle_file:
        pickle.dump(grid_search.cv_results_, pickle_file)
