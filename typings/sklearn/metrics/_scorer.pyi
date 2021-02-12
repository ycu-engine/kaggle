"""
This type stub file was generated by pyright.
"""

from . import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, explained_variance_score, log_loss, max_error, mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, roc_auc_score, top_k_accuracy_score
from .cluster import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score
from ..utils.validation import _deprecate_positional_args

"""
The :mod:`sklearn.metrics.scorer` submodule implements a flexible
interface for model selection and evaluation using
arbitrary score functions.

A scorer object is a callable that can be passed to
:class:`~sklearn.model_selection.GridSearchCV` or
:func:`sklearn.model_selection.cross_val_score` as the ``scoring``
parameter, to specify how a model should be evaluated.

The signature of the call is ``(estimator, X, y)`` where ``estimator``
is the model to be evaluated, ``X`` is the test data and ``y`` is the
ground truth labeling (or ``None`` in the case of unsupervised models).
"""
class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.
    """
    def __init__(self, **scorers) -> None:
        ...
    
    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        ...
    


class _BaseScorer:
    def __init__(self, score_func, sign, kwargs) -> None:
        ...
    
    def __repr__(self):
        ...
    
    def __call__(self, estimator, X, y_true, sample_weight=...):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        ...
    


class _PredictScorer(_BaseScorer):
    ...


class _ProbaScorer(_BaseScorer):
    ...


class _ThresholdScorer(_BaseScorer):
    ...


def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.

    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    ...

@_deprecate_positional_args
def check_scoring(estimator, scoring=..., *, allow_none=...):
    """Determine scorer from user options.

    A TypeError will be thrown if the estimator cannot be scored.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    allow_none : bool, default=False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.

    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    ...

@_deprecate_positional_args
def make_scorer(score_func, *, greater_is_better=..., needs_proba=..., needs_threshold=..., **kwargs):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a score function, such as :func:`~sklearn.metrics.accuracy_score`,
    :func:`~sklearn.metrics.mean_squared_error`,
    :func:`~sklearn.metrics.adjusted_rand_index` or
    :func:`~sklearn.metrics.average_precision`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : bool, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function`.
    """
    ...

explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error, greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
neg_mean_absolute_percentage_error_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
neg_median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)
neg_root_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
neg_mean_poisson_deviance_scorer = make_scorer(mean_poisson_deviance, greater_is_better=False)
neg_mean_gamma_deviance_scorer = make_scorer(mean_gamma_deviance, greater_is_better=False)
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
top_k_accuracy_scorer = make_scorer(top_k_accuracy_score, greater_is_better=True, needs_threshold=True)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score, needs_threshold=True)
roc_auc_ovo_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo')
roc_auc_ovo_weighted_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo', average='weighted')
roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
roc_auc_ovr_weighted_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
neg_brier_score_scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
rand_scorer = make_scorer(rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)
SCORERS = dict(explained_variance=explained_variance_scorer, r2=r2_scorer, max_error=max_error_scorer, neg_median_absolute_error=neg_median_absolute_error_scorer, neg_mean_absolute_error=neg_mean_absolute_error_scorer, neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer, neg_mean_squared_error=neg_mean_squared_error_scorer, neg_mean_squared_log_error=neg_mean_squared_log_error_scorer, neg_root_mean_squared_error=neg_root_mean_squared_error_scorer, neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer, neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer, accuracy=accuracy_scorer, top_k_accuracy=top_k_accuracy_scorer, roc_auc=roc_auc_scorer, roc_auc_ovr=roc_auc_ovr_scorer, roc_auc_ovo=roc_auc_ovo_scorer, roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer, roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer, balanced_accuracy=balanced_accuracy_scorer, average_precision=average_precision_scorer, neg_log_loss=neg_log_loss_scorer, neg_brier_score=neg_brier_score_scorer, adjusted_rand_score=adjusted_rand_scorer, rand_score=rand_scorer, homogeneity_score=homogeneity_scorer, completeness_score=completeness_scorer, v_measure_score=v_measure_scorer, mutual_info_score=mutual_info_scorer, adjusted_mutual_info_score=adjusted_mutual_info_scorer, normalized_mutual_info_score=normalized_mutual_info_scorer, fowlkes_mallows_score=fowlkes_mallows_scorer)
