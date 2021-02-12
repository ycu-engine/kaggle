"""
This type stub file was generated by pyright.
"""

import numpy as np
from ._base import LinearModel
from ..base import RegressorMixin
from ..utils.validation import _deprecate_positional_args

"""
A Theil-Sen Estimator for Multiple Linear Regression Model
"""
_EPSILON = np.finfo(np.double).eps
class TheilSenRegressor(RegressorMixin, LinearModel):
    """Theil-Sen Estimator: robust multivariate regression model.

    The algorithm calculates least square solutions on subsets with size
    n_subsamples of the samples in X. Any value of n_subsamples between the
    number of features and samples leads to an estimator with a compromise
    between robustness and efficiency. Since the number of least square
    solutions is "n_samples choose n_subsamples", it can be extremely large
    and can therefore be limited with max_subpopulation. If this limit is
    reached, the subsets are chosen randomly. In a final step, the spatial
    median (or L1 median) is calculated of all least square solutions.

    Read more in the :ref:`User Guide <theil_sen_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    max_subpopulation : int, default=1e4
        Instead of computing with a set of cardinality 'n choose k', where n is
        the number of samples and k is the number of subsamples (at least
        number of features), consider only a stochastic subpopulation of a
        given maximal size if 'n choose k' is larger than max_subpopulation.
        For other than small problem sizes this parameter will determine
        memory usage and runtime if n_subsamples is not changed.

    n_subsamples : int, default=None
        Number of samples to calculate the parameters. This is at least the
        number of features (plus 1 if fit_intercept=True) and the number of
        samples as a maximum. A lower number leads to a higher breakdown
        point and a low efficiency while a high number leads to a low
        breakdown point and a high efficiency. If None, take the
        minimum number of subsamples leading to maximal robustness.
        If n_subsamples is set to n_samples, Theil-Sen is identical to least
        squares.

    max_iter : int, default=300
        Maximum number of iterations for the calculation of spatial median.

    tol : float, default=1.e-3
        Tolerance when calculating spatial median.

    random_state : int, RandomState instance or None, default=None
        A random number generator instance to define the state of the random
        permutations generator. Pass an int for reproducible output across
        multiple function calls.
        See :term:`Glossary <random_state>`

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the regression model (median of distribution).

    intercept_ : float
        Estimated intercept of regression model.

    breakdown_ : float
        Approximated breakdown point.

    n_iter_ : int
        Number of iterations needed for the spatial median.

    n_subpopulation_ : int
        Number of combinations taken into account from 'n choose k', where n is
        the number of samples and k is the number of subsamples.

    Examples
    --------
    >>> from sklearn.linear_model import TheilSenRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, random_state=0)
    >>> reg = TheilSenRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.9884...
    >>> reg.predict(X[:1,])
    array([-31.5871...])

    References
    ----------
    - Theil-Sen Estimators in a Multiple Linear Regression Model, 2009
      Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
      http://home.olemiss.edu/~xdang/papers/MTSE.pdf
    """
    @_deprecate_positional_args
    def __init__(self, *, fit_intercept=..., copy_X=..., max_subpopulation=..., n_subsamples=..., max_iter=..., tol=..., random_state=..., n_jobs=..., verbose=...) -> None:
        ...
    
    def fit(self, X, y):
        """Fit linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        ...
    


