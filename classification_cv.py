from classification import *
import json

if __name__ == '__main__':
    predictors, y_h, columns = create_data(2)
    loo_search = cv_leave_one_out_encoder(predictors, y_h, columns)
    with open("studies/loo_classification_cv.json", 'w') as json_file:
        json.dump(loo_search, json_file, indent=4)

    search = cv_sampling(predictors, y_h, columns)
    with open("studies/classification_cv.json", 'w') as json_file:
        json.dump(search, json_file, indent=4)
