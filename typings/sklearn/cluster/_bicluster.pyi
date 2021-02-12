"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator, BiclusterMixin
from ..utils.validation import _deprecate_positional_args

"""Spectral biclustering algorithms."""
class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for spectral biclustering."""
    @abstractmethod
    def __init__(self, n_clusters=..., svd_method=..., n_svd_vecs=..., mini_batch=..., init=..., n_init=..., n_jobs=..., random_state=...) -> None:
        ...
    
    def fit(self, X, y=...):
        """Creates a biclustering for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : Ignored

        """
        ...
    


class SpectralCoclustering(BaseSpectral):
    """Spectral Co-Clustering algorithm (Dhillon, 2001).

    Clusters rows and columns of an array `X` to solve the relaxed
    normalized cut of the bipartite graph created from `X` as follows:
    the edge between row vertex `i` and column vertex `j` has weight
    `X[i, j]`.

    The resulting bicluster structure is block-diagonal, since each
    row and each column belongs to exactly one bicluster.

    Supports sparse matrices, as long as they are nonnegative.

    Read more in the :ref:`User Guide <spectral_coclustering>`.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of biclusters to find.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', use
        :func:`sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', use
        :func:`scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random', or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. deprecated:: 0.23
            ``n_jobs`` was deprecated in version 0.23 and will be removed in
            1.0 (renaming of 0.25).

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        The bicluster label of each row.

    column_labels_ : array-like of shape (n_cols,)
        The bicluster label of each column.

    Examples
    --------
    >>> from sklearn.cluster import SpectralCoclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_ #doctest: +SKIP
    array([0, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_ #doctest: +SKIP
    array([0, 0], dtype=int32)
    >>> clustering
    SpectralCoclustering(n_clusters=2, random_state=0)

    References
    ----------

    * Dhillon, Inderjit S, 2001. `Co-clustering documents and words using
      bipartite spectral graph partitioning
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011>`__.

    """
    @_deprecate_positional_args
    def __init__(self, n_clusters=..., *, svd_method=..., n_svd_vecs=..., mini_batch=..., init=..., n_init=..., n_jobs=..., random_state=...) -> None:
        ...
    


class SpectralBiclustering(BaseSpectral):
    """Spectral biclustering (Kluger, 2003).

    Partitions rows and columns under the assumption that the data has
    an underlying checkerboard structure. For instance, if there are
    two row partitions and three column partitions, each row will
    belong to three biclusters, and each column will belong to two
    biclusters. The outer product of the corresponding row and column
    label vectors gives this checkerboard structure.

    Read more in the :ref:`User Guide <spectral_biclustering>`.

    Parameters
    ----------
    n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
        The number of row and column clusters in the checkerboard
        structure.

    method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
        Method of normalizing and converting singular vectors into
        biclusters. May be one of 'scale', 'bistochastic', or 'log'.
        The authors recommend using 'log'. If the data is sparse,
        however, log normalization will not work, which is why the
        default is 'bistochastic'.

        .. warning::
           if `method='log'`, the data must be sparse.

    n_components : int, default=6
        Number of singular vectors to check.

    n_best : int, default=3
        Number of best singular vectors to which to project the data
        for clustering.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', uses
        :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', uses
        `scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'} or ndarray of (n_clusters, n_features), \
            default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. deprecated:: 0.23
            ``n_jobs`` was deprecated in version 0.23 and will be removed in
            1.0 (renaming of 0.25).

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        Row partition labels.

    column_labels_ : array-like of shape (n_cols,)
        Column partition labels.

    Examples
    --------
    >>> from sklearn.cluster import SpectralBiclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([0, 1], dtype=int32)
    >>> clustering
    SpectralBiclustering(n_clusters=2, random_state=0)

    References
    ----------

    * Kluger, Yuval, et. al., 2003. `Spectral biclustering of microarray
      data: coclustering genes and conditions
      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.1608>`__.

    """
    @_deprecate_positional_args
    def __init__(self, n_clusters=..., *, method=..., n_components=..., n_best=..., svd_method=..., n_svd_vecs=..., mini_batch=..., init=..., n_init=..., n_jobs=..., random_state=...) -> None:
        ...
    


