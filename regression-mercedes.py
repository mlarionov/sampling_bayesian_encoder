import optuna
import pandas as pd
from category_encoders import SamplingBayesianEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.sampling_bayesian import EncoderWrapper, TaskType
from optuna.distributions import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils import dump_optuna_results


def run_experiments():
    all_data = pd.read_csv('train.csv.zip')

    # Removing duplicate or constant columns
    # as per https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554
    columns_to_remove = ['X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347',
                         'X382', 'X232', 'X279', 'X35', 'X37', 'X39', 'X302', 'X113', 'X134', 'X147', 'X222',
                         'X102', 'X214', 'X239', 'X76', 'X324', 'X248', 'X253', 'X385', 'X172', 'X216', 'X213',
                         'X84', 'X244', 'X122', 'X243', 'X320', 'X245', 'X94', 'X242', 'X199', 'X119', 'X227',
                         'X146', 'X226', 'X326', 'X360', 'X262', 'X266', 'X247', 'X254', 'X364', 'X365', 'X296', 'X299',
                         'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']
    new_columns = [col for col in all_data.columns if col not in columns_to_remove]
    data1 = all_data[new_columns]

    # Reading test data
    test_data = pd.read_csv('test.csv.zip')

    # Training the label encoder
    combined = pd.concat([all_data, test_data], axis=0, sort=False)
    cat_column_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
    label_encoders = {}
    for col in cat_column_names:
        label_encoder = LabelEncoder()
        label_encoder.fit(combined[col])
        label_encoders[col] = label_encoder

    # Label encoding training data
    data2 = data1.copy()
    mean_enc_columns = [data2.columns.get_loc(c) for c in data2.columns if c in cat_column_names]
    for col in cat_column_names:
        data2[col] = label_encoders[col].transform(data2[col])

    X = data2.iloc[:, 2:].values
    y = data2.iloc[:, 1].values

    # Kaggle data set.
    kaggle1 = test_data[new_columns[2:]].copy()
    for col in cat_column_names:
        kaggle1[col] = label_encoders[col].transform(kaggle1[col])
    X_kaggle = kaggle1.values

    # ## Cross validation
    #
    # We will do cross-validation to ensure validity of comparison between the two approaches
    mean_enc_column_names = [c for c in data2.columns if c in cat_column_names]

    # ### Leave one out encoding
    study_loo(X, X_kaggle, data1, mean_enc_column_names, test_data, y)

    # Sampling Bayesian Encoder
    sbe_cv(X, X_kaggle, data1, mean_enc_column_names, test_data, y)


def sbe_cv(X, X_kaggle, data1, mean_enc_column_names, test_data, y):
    pte = SamplingBayesianEncoder(cols=mean_enc_column_names, random_state=2834, task=TaskType.REGRESSION)
    model = RandomForestRegressor(n_estimators=400, random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=data1.columns[2:])
    X_test = pd.DataFrame(X_test, columns=data1.columns[2:])

    params_squared = lambda x: (x[0], x[0] ** 2)
    params_squared.__str__ = lambda: 'lambda x: (x[0], x[0] ** 2)'

    param_distribution = {
        'encoder__prior_samples_ratio': LogUniformDistribution(1E-10, 1E-5),
        'encoder__n_draws': IntUniformDistribution(1, 40),
        'encoder__mapper': CategoricalDistribution(['mean', 'identity', params_squared]),
        'estimator__max_depth': IntUniformDistribution(2, 40),
        'estimator__max_features': IntUniformDistribution(1, X_test.shape[1]),
        'estimator__min_samples_leaf': IntUniformDistribution(1, 15)
    }
    search = optuna.integration.OptunaSearchCV(wrapper_model, param_distribution,
                                               cv=5, n_jobs=-1, random_state=514, n_trials=40, timeout=None,
                                               scoring='r2')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    test_predict = search.best_estimator_.predict(X_test)
    test_score = r2_score(y_test, test_predict)
    print('Test R^2: ', test_score)
    # In[34]:
    preds_kaggle = search.best_estimator_.predict(pd.DataFrame(X_kaggle, columns=data1.columns[2:]))
    preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
    preds_kaggle_df.head(2)
    preds_kaggle_df.to_csv('sbe_submission.csv', index=False)
    # Saving CV results:
    with open("studies/regression_sbe_cv.json", 'w') as json_file:
        json.dump(dump_optuna_results(search, test_score, search.best_estimator_.estimator.feature_importances_),
                  json_file, indent=4,  default=lambda obj: obj.__str__())


def study_loo(X, X_kaggle, data1, mean_enc_column_names, test_data, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=data1.columns[2:])
    X_test = pd.DataFrame(X_test, columns=data1.columns[2:])
    loo = LeaveOneOutEncoder(cols=mean_enc_column_names, random_state=2834)
    rf = RandomForestRegressor(n_estimators=400, random_state=2834, n_jobs=-1)
    pipe = Pipeline(steps=[('loo', loo), ('rf', rf)])
    param_distribution = {
        'loo__sigma': LogUniformDistribution(1E-5, 1E-1),
        'rf__max_depth': IntUniformDistribution(2, 40),
        'rf__max_features': IntUniformDistribution(1, X_test.shape[1]),
        'rf__min_samples_leaf': IntUniformDistribution(1, 15)
    }
    search = optuna.integration.OptunaSearchCV(pipe, param_distribution,
                                               cv=5, n_jobs=-1, random_state=514, n_trials=40, timeout=None,
                                               scoring='r2')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    test_predict = search.best_estimator_.predict(X_test)
    test_score = r2_score(y_test, test_predict)
    print('Test R^2: ', test_score)
    # In[32]:
    preds_kaggle = search.best_estimator_.predict(pd.DataFrame(X_kaggle, columns=data1.columns[2:]))
    preds_kaggle_df = pd.DataFrame({'ID': test_data.ID, 'y': preds_kaggle, })
    preds_kaggle_df.head(2)
    preds_kaggle_df.to_csv('loo_submission.csv', index=False)
    # Saving CV results:
    with open("studies/regression_loo_cv.json", 'w') as json_file:
        json.dump(dump_optuna_results(search, test_score,
                                      search.best_estimator_.named_steps['rf'].feature_importances_), json_file,
                  indent=4)
    return X_test, test_score


if __name__ == '__main__':
    run_experiments()
