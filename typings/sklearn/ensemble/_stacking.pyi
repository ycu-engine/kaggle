"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod
from ..base import ClassifierMixin, RegressorMixin, TransformerMixin
from ._base import _BaseHeterogeneousEnsemble
from ..utils.metaestimators import if_delegate_has_method
from ..utils.validation import _deprecate_positional_args

"""Stacking classifier and regressor."""
class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""
    @abstractmethod
    def __init__(self, estimators, final_estimator=..., *, cv=..., stack_method=..., n_jobs=..., verbose=..., passthrough=...) -> None:
        ...
    
    def fit(self, X, y, sample_weight=...):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """
        ...
    
    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        ...
    
    @if_delegate_has_method(delegate='final_estimator_')
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        ...
    


class StackingClassifier(ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, \
            default='auto'
        Methods called for each base estimator. It can be:

        * if 'auto', it will try to invoke, for each estimator,
          `'predict_proba'`, `'decision_function'` or `'predict'` in that
          order.
        * otherwise, one of `'predict_proba'`, `'decision_function'` or
          `'predict'`. If the method is not implemented by the estimator, it
          will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel all `estimators` `fit`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See Glossary for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    final_estimator_ : estimator
        The classifier which predicts given the output of `estimators_`.

    stack_method_ : list of str
        The method used by each base estimator.

    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    The first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...

    """
    @_deprecate_positional_args
    def __init__(self, estimators, final_estimator=..., *, cv=..., stack_method=..., n_jobs=..., passthrough=..., verbose=...) -> None:
        ...
    
    def fit(self, X, y, sample_weight=...):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        ...
    
    @if_delegate_has_method(delegate='final_estimator_')
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        ...
    
    @if_delegate_has_method(delegate='final_estimator_')
    def predict_proba(self, X):
        """Predict class probabilities for X using
        `final_estimator_.predict_proba`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        ...
    
    @if_delegate_has_method(delegate='final_estimator_')
    def decision_function(self, X):
        """Predict decision function for samples in X using
        `final_estimator_.decision_function`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        decisions : ndarray of shape (n_samples,), (n_samples, n_classes), \
            or (n_samples, n_classes * (n_classes-1) / 2)
            The decision function computed the final estimator.
        """
        ...
    
    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        ...
    


class StackingRegressor(RegressorMixin, _BaseStacking):
    """Stack of estimators with a final regressor.

    Stacked generalization consists in stacking the output of individual
    estimator and use a regressor to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

    final_estimator : estimator, default=None
        A regressor which will be used to combine the base estimators.
        The default regressor is a :class:`~sklearn.linear_model.RidgeCV`.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See Glossary for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    estimators_ : list of estimator
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.


    final_estimator_ : estimator
        The regressor to stacked the base estimators fitted.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> from sklearn.svm import LinearSVR
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import StackingRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> estimators = [
    ...     ('lr', RidgeCV()),
    ...     ('svr', LinearSVR(random_state=42))
    ... ]
    >>> reg = StackingRegressor(
    ...     estimators=estimators,
    ...     final_estimator=RandomForestRegressor(n_estimators=10,
    ...                                           random_state=42)
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> reg.fit(X_train, y_train).score(X_test, y_test)
    0.3...

    """
    @_deprecate_positional_args
    def __init__(self, estimators, final_estimator=..., *, cv=..., n_jobs=..., passthrough=..., verbose=...) -> None:
        ...
    
    def fit(self, X, y, sample_weight=...):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        ...
    
    def transform(self, X):
        """Return the predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators)
            Prediction outputs for each estimator.
        """
        ...
    


