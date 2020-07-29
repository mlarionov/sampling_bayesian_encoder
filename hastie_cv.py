from classification import *
import json
if __name__ == '__main__':
    predictors, y_h = create_data_hastie()
    loo_search = cv_leave_one_out_encoder(predictors, y_h)
    with open("loo_hastie_cv.json", 'w') as json_file:
        json.dump(loo_search, json_file)

    search = cv_sampling(predictors, y_h)
    with open("hastie_cv.json", 'w') as json_file:
        json.dump(search, json_file)
