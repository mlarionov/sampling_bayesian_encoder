from optuna.integration import OptunaSearchCV


def dump_optuna_results(search: OptunaSearchCV, test_score, feature_importance=''):
    return {"val_score": search.best_score_,
            "best_params": search.best_params_,
            "test_score": test_score,
            "feature_importance": feature_importance.tolist()}


def get_serializable_cv_results(cv_results: dict):
    return {key: (value.tolist() if hasattr(value, 'tolist') else value) for (key, value) in cv_results.items()}
