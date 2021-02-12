"""
This type stub file was generated by pyright.
"""

from . import EmpiricalCovariance
from ..utils.validation import _deprecate_positional_args
from ..utils.deprecation import deprecated

"""GraphicalLasso: sparse inverse covariance estimation with an l1-penalized
estimator.
"""
def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        The sample covariance matrix.

    Notes
    -----
    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.
    """
    ...

@_deprecate_positional_args
def graphical_lasso(emp_cov, alpha, *, cov_init=..., mode=..., tol=..., enet_tol=..., max_iter=..., verbose=..., return_costs=..., eps=..., return_n_iter=...):
    """l1-penalized covariance estimator

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        graph_lasso has been renamed to graphical_lasso

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance. If None, then the empirical
        covariance is used.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : bool, default=Flase
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        The estimated covariance matrix.

    precision : ndarray of shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso, GraphicalLassoCV

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.
    """
    ...

class GraphicalLasso(EmpiricalCovariance):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLasso has been renamed to GraphicalLasso

    Parameters
    ----------
    alpha : float, default=0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLasso
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLasso().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.049, 0.218, 0.019],
           [0.049, 0.364, 0.017, 0.034],
           [0.218, 0.017, 0.322, 0.093],
           [0.019, 0.034, 0.093, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])

    See Also
    --------
    graphical_lasso, GraphicalLassoCV
    """
    @_deprecate_positional_args
    def __init__(self, alpha=..., *, mode=..., tol=..., enet_tol=..., max_iter=..., verbose=..., assume_centered=...) -> None:
        ...
    
    def fit(self, X, y=...):
        """Fits the GraphicalLasso model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        ...
    


def graphical_lasso_path(X, alphas, cov_init=..., X_test=..., mode=..., tol=..., enet_tol=..., max_iter=..., verbose=...):
    """l1-penalized covariance estimator along a path of decreasing alphas

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    alphas : array-like of shape (n_alphas,)
        The list of regularization parameters, decreasing order.

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance.

    X_test : array of shape (n_test_samples, n_features), default=None
        Optional test matrix to measure generalisation error.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. The tolerance must be a positive
        number.

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. The tolerance must be a positive number.

    max_iter : int, default=100
        The maximum number of iterations. This parameter should be a strictly
        positive integer.

    verbose : int or bool, default=False
        The higher the verbosity flag, the more information is printed
        during the fitting.

    Returns
    -------
    covariances_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated covariance matrices.

    precisions_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated (sparse) precision matrices.

    scores_ : list of shape (n_alphas,), dtype=float
        The generalisation error (log-likelihood) on the test data.
        Returned only if test data is passed.
    """
    ...

class GraphicalLassoCV(GraphicalLasso):
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLassoCV has been renamed to GraphicalLassoCV

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is (0, inf] when floats given.

    n_refinements : int, default=4
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed. Range is [1, inf).

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None changed from 3-fold to 5-fold.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        Maximum number of iterations.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where number of features is greater
        than number of samples. Elsewhere prefer cd which is more numerically
        stable.

    n_jobs : int, default=None
        number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    verbose : bool, default=False
        If verbose is True, the objective function and duality gap are
        printed at each iteration.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix (inverse covariance).

    alpha_ : float
        Penalization parameter selected.

    cv_alphas_ : list of shape (n_alphas,), dtype=float
        All penalization parameters explored.

        .. deprecated:: 0.24
            The `cv_alphas_` attribute is deprecated in version 0.24 in favor
            of `cv_results_['alphas']` and will be removed in version
            1.1 (renaming of 0.26).

    grid_scores_ : ndarray of shape (n_alphas, n_folds)
        Log-likelihood score on left-out data across folds.

        .. deprecated:: 0.24
            The `grid_scores_` attribute is deprecated in version 0.24 in favor
            of `cv_results_` and will be removed in version
            1.1 (renaming of 0.26).

    cv_results_ : dict of ndarrays
        A dict with keys:

        alphas : ndarray of shape (n_alphas,)
            All penalization parameters explored.

        split(k)_score : ndarray of shape (n_alphas,)
            Log-likelihood score on left-out data across (k)th fold.

        mean_score : ndarray of shape (n_alphas,)
            Mean of scores over the folds.

        std_score : ndarray of shape (n_alphas,)
            Standard deviation of scores over the folds.

        .. versionadded:: 0.24

    n_iter_ : int
        Number of iterations run for the optimal alpha.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLassoCV
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLassoCV().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.051, 0.22 , 0.017],
           [0.051, 0.364, 0.018, 0.036],
           [0.22 , 0.018, 0.322, 0.094],
           [0.017, 0.036, 0.094, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])

    See Also
    --------
    graphical_lasso, GraphicalLasso

    Notes
    -----
    The search for the optimal penalization parameter (alpha) is done on an
    iteratively refined grid: first the cross-validated scores on a grid are
    computed, then a new refined grid is centered around the maximum, and so
    on.

    One of the challenges which is faced here is that the solvers can
    fail to converge to a well-conditioned estimate. The corresponding
    values of alpha then come out as missing values, but the optimum may
    be close to these missing values.
    """
    @_deprecate_positional_args
    def __init__(self, *, alphas=..., n_refinements=..., cv=..., tol=..., enet_tol=..., max_iter=..., mode=..., n_jobs=..., verbose=..., assume_centered=...) -> None:
        ...
    
    def fit(self, X, y=...):
        """Fits the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        ...
    
    @deprecated("The grid_scores_ attribute is deprecated in version 0.24 in favor " "of cv_results_ and will be removed in version 1.1 (renaming of 0.26).")
    @property
    def grid_scores_(self):
        ...
    
    @deprecated("The cv_alphas_ attribute is deprecated in version 0.24 in favor " "of cv_results_['alpha'] and will be removed in version 1.1 " "(renaming of 0.26).")
    @property
    def cv_alphas_(self):
        ...
    


