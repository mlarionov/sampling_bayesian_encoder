from classification import *
from category_encoders import sampling_bayesian
import pickle

if __name__ == '__main__':
    predictors, y_h = create_data()

    with open("classification_cv.pickle", 'rb') as pickle_file:
        search = pickle.load(pickle_file)

    study_mapper(predictors, y_h, search, "study_mapper.pickle")
    study_n_draws(predictors, y_h, search, "study_n_draws.pickle")
    study_prior_samples_ratio(predictors, y_h, search, "study_prior_samples_ratio.pickle")
    print(search.best_estimator_.estimator.feature_importances)
