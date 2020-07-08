from enum import Enum
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'Michael Larionov'


class TaskType(Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3

    @staticmethod
    def create_accumulator(task):
        if task == TaskType.REGRESSION:
            return NormalGammaAccumulator
        elif task == TaskType.BINARY_CLASSIFICATION:
            return BetaAccumulator


class SamplingBayesianEncoder(BaseEstimator, TransformerMixin):
    """Sampling Bayesian Encoder

    This is a version of target encoder, which learns posterior distribution during training, then takes
    a sample from the distribution during prediction

    References
    ----------

    .. [1] Michael Larionov, Sampling Techniques in Bayesian Target Encoding, arXiv:2006.01317
    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None,
                 prior_samples_ratio=1E-4, n_draws=10, mapper='identity', task=TaskType.BINARY_CLASSIFICATION):
        """
        :param verbose: Level of verbosity. Default: 0
        :param cols: Categorical columns to be encoded
        :param drop_invariant: Drop columns that have the same value. Default: False
        :param return_df: If True return DataFrame, even if the input is not. Default: True
        :param handle_unknown: How to handle unknown. If 'value' then use prior distribution. If return_nan then  None
        :param handle_missing: How to handle missing. If 'value' then use prior distribution. If return_nan then  None
        :param random_state: Random state is used when taking samples
        :param prior_samples_ratio: Degree of the influence of the prior distribution
        :param n_draws: Number of draws (sample size)
        :param mapper: string or callable: Mapper to be used. Default: identity
        :param task: TaskType.
        """
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.random_state = random_state
        self.prior_samples_ratio = prior_samples_ratio
        self.feature_names = None
        self.n_draws = n_draws
        self.mapper = mapper
        self.task = task

    def fit(self, X, y):
        """Fit encoder according to X and binary y.

            :param X: array-like, shape = [n_samples, n_features]
               Training vectors, where n_samples is the number of samples
               and n_features is the number of features.
            :param y: array-like, shape = [n_samples]
               Binary target values.
            :return: self

        """

        # Unite parameters into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                          "Not found in generated cols.\n{}".format(e))
        return self

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        When the data are used for model training, it is important to also pass the target in order to apply leave
        one out.


            :rtype: array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
            :param X: array-like, shape = [n_samples, n_features]
            :param y: array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)
            :param override_return_df:

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas types
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in DataFrame')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
            transform(X, y)
        and not with:
            transform(X)
        """

        # the interface requires 'y=None' in the signature but we need 'y'
        if y is None:
            raise (TypeError, 'fit_transform() missing argument: ''y''')

        return self.fit(X, y).transform(X, y)

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self.accumulator = TaskType.create_accumulator(self.task)(y, self.prior_samples_ratio)
        prior = self.accumulator.prior

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            estimate = self.accumulator.get_posterior_parameters(X, col)

            # Deal with special cases
            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            singles = estimate[-1].isnull()
            for param_index in range(len(prior)):
                estimate[param_index][singles] = prior[param_index]

                if self.handle_unknown == 'return_nan':
                    estimate[param_index].loc[-1] = np.nan
                elif self.handle_unknown == 'value':
                    estimate[param_index].loc[-1] = prior[param_index]

                if self.handle_missing == 'return_nan':
                    estimate[param_index].loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == 'value':
                    estimate[param_index].loc[-2] = prior[param_index]

            # Store the m-probability estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _score_one_draw(self, X_in: pd.DataFrame):
        X = X_in.copy(deep=True)
        sample_function = np.vectorize(self.accumulator.sample_single)  # , signature='(a1),(a2),(a3),(a4)->(k)')
        for col in self.cols:
            sample_results = sample_function(*self.mapping[col])
            mapper = Mapping.create_mapper(self.mapper)
            impute = mapper(sample_results)
            if type(impute) is not tuple:
                impute = (impute,)
            columns = [f"{col}_encoded_{i}" for i in range(len(impute))]
            data = np.vstack(impute).T if len(columns) > 1 else impute[0]
            sample_results_df = pd.DataFrame(data=data, columns=columns,
                                             index=self.mapping[col][0].index)
            for column in columns:
                X[column] = X[col].map(sample_results_df[column])
            X = X.drop(columns=[col])
        return X

    def _score(self, X):
        return pd.concat([self._score_one_draw(X) for _ in range(self.n_draws)])

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

    def expand_y(self, y):
        """
        Transform the dependent variable so that it matches the size of the oversampled array
        :param y: dependent variable
        :return: dependent variable duplicated n_draws times
        """
        y = np.array(y).reshape(-1)
        return np.hstack([y for _ in range(self.n_draws)])

    def average_y(self, y):
        """
        Average the values of the oversampled dependent variable to reduce its size to the original size.
        This function is opposite to expand_y()
        :param y: oversampled dependent variable
        :return: the averaged dependent variable of the original size
        """
        split_y = np.split(y, self.n_draws)
        split_y_combined = np.vstack(split_y)
        return split_y_combined.mean(axis=0)


class EncoderWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Wraps encoder and estimator, orchestrates fit() and predict() pipelines.
    Works for regression and classification.
    """

    def __init__(self, encoder, estimator):
        """
        :param encoder: Encoder instance
        :param estimator: Estimator instance
        """
        self.encoder = encoder
        self.estimator = estimator

    def fit(self, X, y):
        """
        Fit the wrapper model. First fits the encoder model, then transforms the data and fits the estimator model
        :param X: predictor variable
        :param y: dependent variable
        :return: self
        """
        self.encoder.fit(X, y)
        X_transformed = self.encoder.transform(X)
        y_transformed = self.encoder.expand_y(y)
        self.estimator.fit(X_transformed, y_transformed)
        return self

    def predict_proba(self, X):
        """
        Predict probability when it is supported by the estimator.
        This is expected for the binary classification models
        :param X: predictor variables
        :return: array of probabilities for the positive class only
        """
        assert hasattr(self.estimator, 'predict_proba'), '''
            predict_proba() method is not available. You may be dealing with a Regression case 
        '''
        X_transformed = self.encoder.transform(X)
        preds = self.estimator.predict_proba(X_transformed)[:, 1]
        return self.encoder.average_y(preds)

    def predict(self, X):
        """
        Predicts the values. If the estimator supports predicting probabilities (binary classification) use that
        :param X: the predictor variables
        :return: the predictions.
        """
        if hasattr(self.estimator, 'predict_proba'):
            return self.predict_proba(X).round()
        else:
            X_transformed = self.encoder.transform(X)
            preds = self.estimator.predict(X_transformed)
            return self.encoder.average_y(preds)


class Mapping(object):
    """
    Mapping functions
    """

    @staticmethod
    def create_mapper(mapper) -> Callable:
        """
        If an argument is a function, returns it. Otherwise creates the function by name.
        :param mapper: A Callable or a mapper name
        :return: a Callable
        """
        if callable(mapper):
            return mapper
        elif mapper == 'mean':
            return Mapping.mean
        elif mapper == 'identity':
            return Mapping.identity
        elif mapper == 'weight_of_evidence':
            return Mapping.weight_of_evidence
        else:
            raise ValueError('Unknown mapper: ', mapper)

    @staticmethod
    def identity(sample_results: Tuple) -> Tuple:
        """
        Identity function
        """
        return sample_results

    @staticmethod
    def mean(sample_results: Tuple) -> Tuple:
        """
        Identity function
        """
        return sample_results[0],

    @staticmethod
    def weight_of_evidence(sample_results: Tuple) -> Tuple:
        """
        Weight of evidence
        :param sample_results:
        :return:
        """
        p = sample_results[0]
        return np.log(p/(1-p))


class NormalGammaAccumulator(object):
    """
    Accumulator for Normal-Gamma distribution. Computes the parameters of the posterior distribution. Samples
    from the distribution
    """

    def __init__(self, y, prior_samples_ratio: float):
        """
        :param y: The dependent variable
        :param prior_samples_ratio: indicates the degree of influence of the prior distribution
        """
        self.y = y
        self.prior = self._compute_posterior_parameters(y.mean(), y.var(), y.shape[0])
        self.prior_samples_ratio = prior_samples_ratio

    @staticmethod
    def _compute_posterior_parameters(y_bar, y_var, n, mu_0=0, nu=0, alpha=0, beta=0) -> Tuple:
        ss = y_var * (n - 1)
        new_mu = (nu * mu_0 + n * y_bar) / (nu + n)
        new_nu = nu + n
        new_alpha = alpha + n / 2
        new_beta = beta + 1 / 2 * ss + n * nu / (n + nu) * (y_bar - mu_0) ** 2 / 2
        return new_mu, new_nu, new_alpha, new_beta

    def get_posterior_parameters(self, X, col: str) -> Tuple:
        """
        Computes the parameters of the posterior distribution for a column
        :param X: predictor variable
        :param col: column name
        :return: a tuple of the parameters of the posterior distribution
        """
        stats = self.y.groupby(X[col]).agg(['mean', 'count', 'var'])
        return self._compute_posterior_parameters(stats['mean'], stats['var'], stats['count'],
                                                  self.prior[0], self.prior[1] * self.prior_samples_ratio,
                                                  self.prior[2] * self.prior_samples_ratio,
                                                  self.prior[3] * self.prior_samples_ratio)

    @staticmethod
    def sample_single(mu, lambda_, alpha, beta) -> Tuple:
        """
        Generate a sample from the posterior distribution
        :param mu: mu
        :param lambda_: lambda
        :param alpha: alpha
        :param beta: beta
        :return: a tuple that returns $\mu$ and $\sigma^2$.
        """
        shape = alpha
        scale = 1 / beta
        tau = np.random.gamma(shape, scale)
        x = np.random.normal(mu, 1 / np.sqrt(lambda_ * tau))
        sigma_2 = 1 / tau
        return x, sigma_2


class BetaAccumulator(object):
    """
    Accumulator for Normal-Gamma distribution. Computes the parameters of the posterior distribution. Samples
    from the distribution
    """

    def __init__(self, y, prior_samples_ratio: float):
        """
        :param y: The dependent variable
        :param prior_samples_ratio: indicates the degree of influence of the prior distribution
        """
        self.y = y
        self.prior = self._compute_posterior_parameters(y.sum(), y.count() - y.sum())
        self.prior_samples_ratio = prior_samples_ratio

    @staticmethod
    def _compute_posterior_parameters(successes, failures, prior_successes=0, prior_failures=0):
        return prior_successes + successes, prior_failures + failures

    def get_posterior_parameters(self, X, col: str) -> Tuple:
        """
        Computes the parameters of the posterior distribution for a column
        :param X: predictor variable
        :param col: column name
        :return: a tuple of the parameters of the posterior distribution
        """
        stats = self.y.groupby(X[col]).agg(['sum', 'count'])
        prior_successes, prior_failures = self.prior
        return self._compute_posterior_parameters(stats['sum'], stats['count'] - stats['sum'],
                                                  prior_successes * self.prior_samples_ratio,
                                                  prior_failures * self.prior_samples_ratio)

    @staticmethod
    def sample_single(successes, failures) -> float:
        """
        Generate a sample from the posterior distribution
        :param failures:
        :param successes:
        :return: a tuple with one element, which is a sample from beta distribution
        """
        return np.random.beta(successes+1, failures+1)
