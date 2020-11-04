from classification import *
import json
if __name__ == '__main__':
    predictors, y_h, cols = create_data_hastie(9)
    loo_search = cv_leave_one_out_encoder(predictors, y_h, cols)
    with open("studies/loo_hastie_cv.json", 'w') as json_file:
        json.dump(loo_search, json_file, indent=4)

    search = cv_sampling(predictors, y_h, cols)
    with open("studies/hastie_cv.json", 'w') as json_file:
        json.dump(search, json_file, indent=4)
