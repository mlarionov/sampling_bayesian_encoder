from sklearn.model_selection import KFold

from classification import *
import json

if __name__ == '__main__':
    generator = KFold(n_splits=5, shuffle=True, random_state=8506)

    predictors, y_h, cat_column_names = create_data(9)
    with open("studies/classification_cv.json", 'r') as json_file:
        search = json.load(json_file)

    study_mapper(predictors, y_h, search, "studies/study_mapper.json", cat_column_names, generator)
    study_n_draws(predictors, y_h, search, "studies/study_n_draws.json", cat_column_names, generator)
    study_prior_samples_ratio(predictors, y_h, search, "studies/study_prior_samples_ratio.json",
                              cat_column_names, generator)

    predictors, y_h, cat_column_names = create_data_hastie(9)
    with open("studies/hastie_cv.json", 'r') as json_file:
        search = json.load(json_file)

    study_mapper(predictors, y_h, search, "studies/study_mapper_h.json", cat_column_names, generator)
    study_n_draws(predictors, y_h, search, "studies/study_n_draws_h.json", cat_column_names, generator)
    study_prior_samples_ratio(predictors, y_h, search, "studies/study_prior_samples_ratio_h.json",
                              cat_column_names, generator)
