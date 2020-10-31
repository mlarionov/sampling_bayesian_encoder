import optuna
import pandas as pd
from category_encoders import SamplingBayesianEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.sampling_bayesian import EncoderWrapper, TaskType
from optuna.distributions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from utils import dump_optuna_results

def params_squared(x):
    return x[0], x[0] ** 2

def run_experiments():

    continuous_column_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    cat_column_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'native-country']
    all_columns = ['age',
                   'workclass',
                   'fnlwgt',
                   'education',
                   'education-num',
                   'marital-status',
                   'occupation',
                   'relationship',
                   'race',
                   'sex',
                   'capital-gain',
                   'capital-loss',
                   'hours-per-week',
                   'native-country',
                   'class']

    all_data = pd.read_csv('adult_raw.csv', names=all_columns)
    data1 = all_data[all_columns]

    # Reading test data

    # Training the label encoder
    combined = all_data
    label_encoders = {}
    for col in cat_column_names:
        label_encoder = LabelEncoder()
        label_encoder.fit(combined[col])
        label_encoders[col] = label_encoder

    # Label encoding training data
    data2 = data1.copy()
    for col in cat_column_names:
        data2[col] = label_encoders[col].transform(data2[col])

    X = data2[continuous_column_names+cat_column_names].values
    y = (data2['class'].values == ' >50K').astype(int)

    # Leave one out encoding
    study_loo(X, continuous_column_names+cat_column_names, cat_column_names, y)

    # Sampling Bayesian Encoder
    sbe_cv(X, continuous_column_names+cat_column_names, cat_column_names, y)


def sbe_cv(X, train_columns, mean_enc_column_names, y):
    pte = SamplingBayesianEncoder(cols=mean_enc_column_names, random_state=2834, task=TaskType.BINARY_CLASSIFICATION)
    model = RandomForestClassifier(n_estimators=400, random_state=2834, n_jobs=-1)
    wrapper_model = EncoderWrapper(pte, model)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=train_columns)
    X_test = pd.DataFrame(X_test, columns=train_columns)

    param_distribution = {
        'encoder__prior_samples_ratio': LogUniformDistribution(1E-10, 1E-5),
        'encoder__n_draws': IntUniformDistribution(1, 20),
        'encoder__mapper': CategoricalDistribution(['mean', 'weight_of_evidence', params_squared]),
        'estimator__max_depth': IntUniformDistribution(2, 40),
        'estimator__max_features': IntUniformDistribution(1, X_test.shape[1]),
        'estimator__min_samples_leaf': IntUniformDistribution(1, 15)
    }
    search = optuna.integration.OptunaSearchCV(wrapper_model, param_distribution,
                                               cv=5, n_jobs=4, random_state=514, n_trials=40, timeout=None,
                                               scoring='accuracy')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    test_predict = search.best_estimator_.predict(X_test)
    test_score = accuracy_score(y_test, test_predict)
    print('Test accuracy: ', test_score)

    # Saving CV results:
    with open("studies/adult_sbe_cv.json", 'w') as json_file:
        json.dump(dump_optuna_results(search, test_score, search.best_estimator_.estimator.feature_importances_),
                  json_file, indent=4, default=lambda obj: obj.__str__())


def study_loo(X, train_columns, mean_enc_column_names, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2834)
    X_train = pd.DataFrame(X_train, columns=train_columns)
    X_test = pd.DataFrame(X_test, columns=train_columns)
    loo = LeaveOneOutEncoder(cols=mean_enc_column_names, random_state=2834)
    rf = RandomForestClassifier(n_estimators=400, random_state=2834, n_jobs=-1)
    pipe = Pipeline(steps=[('loo', loo), ('rf', rf)])
    param_distribution = {
        'loo__sigma': LogUniformDistribution(1E-5, 1E-1),
        'rf__max_depth': IntUniformDistribution(2, 40),
        'rf__max_features': IntUniformDistribution(1, X_test.shape[1]),
        'rf__min_samples_leaf': IntUniformDistribution(1, 15)
    }
    search = optuna.integration.OptunaSearchCV(pipe, param_distribution,
                                               cv=5, n_jobs=-1, random_state=514, n_trials=40, timeout=None,
                                               scoring='accuracy')
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    test_predict = search.best_estimator_.predict(X_test)
    test_score = accuracy_score(y_test, test_predict)
    print('Test accuracy: ', test_score)

    # Saving CV results:
    with open("studies/adult_loo_cv.json", 'w') as json_file:
        json.dump(dump_optuna_results(search, test_score,
                                      search.best_estimator_.named_steps['rf'].feature_importances_), json_file,
                  indent=4)
    return X_test, test_score


if __name__ == '__main__':
    run_experiments()
