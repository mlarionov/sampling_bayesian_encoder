from classification import *
import json

if __name__ == '__main__':
    predictors, y_h = create_data()

    with open("studies/classification_cv.json", 'r') as json_file:
        search = json.load(json_file)

    study_mapper(predictors, y_h, search, "studies/study_mapper.json")
    study_n_draws(predictors, y_h, search, "studies/study_n_draws.json")
    study_prior_samples_ratio(predictors, y_h, search, "studies/study_prior_samples_ratio.json")

    predictors, y_h = create_data_hastie()
    with open("studies/hastie_cv.json", 'r') as json_file:
        search = json.load(json_file)

    study_mapper(predictors, y_h, search, "studies/study_mapper_h.json")
    study_n_draws(predictors, y_h, search, "studies/study_n_draws_h.json")
    study_prior_samples_ratio(predictors, y_h, search, "studies/study_prior_samples_ratio_h.json")
