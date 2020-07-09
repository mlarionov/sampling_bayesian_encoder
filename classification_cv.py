from classification import *
import pickle

if __name__ == '__main__':
    predictors, y_h = create_data()
    #random_forest(predictors, y_h)
    loo_search = cv_leave_one_out_encoder(predictors, y_h)
    with open("loo_classification_cv.pickle", 'wb') as pickle_file:
        pickle.dump(loo_search, pickle_file)

    search = cv_sampling(predictors, y_h)
    with open("classification_cv.pickle", 'wb') as pickle_file:
        pickle.dump(search, pickle_file)