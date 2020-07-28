from optuna.integration import OptunaSearchCV


def dump_optuna_results(search: OptunaSearchCV, test_score):
    return {"val_score": search.best_score_, "best_params": search.best_params_, "test_score": test_score}
