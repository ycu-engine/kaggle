"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, TransformerMixin
from ..utils.validation import _deprecate_positional_args
from ..utils.deprecation import deprecated

"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""
class _PLS(TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://www.stat.washington.edu/research/reports/2000/tr371.pdf
    """
    @abstractmethod
    def __init__(self, n_components=..., *, scale=..., deflation_mode=..., mode=..., algorithm=..., max_iter=..., tol=..., copy=...) -> None:
        ...
    
    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        """
        ...
    
    def transform(self, X, Y=..., copy=...):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        ...
    
    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        ...
    
    def predict(self, X, copy=...):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        ...
    
    def fit_transform(self, X, y=...):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        ...
    
    @deprecated("Attribute norm_y_weights was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def norm_y_weights(self):
        ...
    
    @deprecated("Attribute x_mean_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def x_mean_(self):
        ...
    
    @deprecated("Attribute y_mean_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def y_mean_(self):
        ...
    
    @deprecated("Attribute x_std_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def x_std_(self):
        ...
    
    @deprecated("Attribute y_std_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def y_std_(self):
        ...
    
    @property
    def x_scores_(self):
        ...
    
    @property
    def y_scores_(self):
        ...
    


class PLSRegression(_PLS):
    """PLS regression

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    PLSRegression()
    >>> Y_pred = pls2.predict(X)
    """
    @_deprecate_positional_args
    def __init__(self, n_components=..., *, scale=..., max_iter=..., tol=..., copy=...) -> None:
        ...
    


class PLSCanonical(_PLS):
    """Partial Least Squares transformer and regressor.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        the maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    PLSCanonical()
    >>> X_c, Y_c = plsca.transform(X, Y)

    See Also
    --------
    CCA
    PLSSVD
    """
    @_deprecate_positional_args
    def __init__(self, n_components=..., *, scale=..., algorithm=..., max_iter=..., tol=..., copy=...) -> None:
        ...
    


class CCA(_PLS):
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        the maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    Examples
    --------
    >>> from sklearn.cross_decomposition import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA(n_components=1)
    >>> cca.fit(X, Y)
    CCA(n_components=1)
    >>> X_c, Y_c = cca.transform(X, Y)

    See Also
    --------
    PLSCanonical
    PLSSVD
    """
    @_deprecate_positional_args
    def __init__(self, n_components=..., *, scale=..., max_iter=..., tol=..., copy=...) -> None:
        ...
    


class PLSSVD(TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the crosscovariance matrix X'Y.
    It is able to project both the training data `X` and the targets `Y`. The
    training data X is projected on the left singular vectors, while the
    targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in `transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in `transform`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, Y)
    >>> X_c, Y_c = pls.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))

    See Also
    --------
    PLSCanonical
    CCA
    """
    @_deprecate_positional_args
    def __init__(self, n_components=..., *, scale=..., copy=...) -> None:
        ...
    
    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets.
        """
        ...
    
    @deprecated("Attribute x_scores_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26). Use est.transform(X) on " "the training data instead.")
    @property
    def x_scores_(self):
        ...
    
    @deprecated("Attribute y_scores_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26). Use est.transform(X, Y) " "on the training data instead.")
    @property
    def y_scores_(self):
        ...
    
    @deprecated("Attribute x_mean_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def x_mean_(self):
        ...
    
    @deprecated("Attribute y_mean_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def y_mean_(self):
        ...
    
    @deprecated("Attribute x_std_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def x_std_(self):
        ...
    
    @deprecated("Attribute y_std_ was deprecated in version 0.24 and " "will be removed in 1.1 (renaming of 0.26).")
    @property
    def y_std_(self):
        ...
    
    def transform(self, X, Y=...):
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y` is not None,
            `(X_transformed, Y_transformed)` otherwise.
        """
        ...
    
    def fit_transform(self, X, y=...):
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y` is not None,
            `(X_transformed, Y_transformed)` otherwise.
        """
        ...
    


