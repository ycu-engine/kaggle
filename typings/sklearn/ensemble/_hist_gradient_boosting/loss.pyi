"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod

"""
This module contains the loss classes.

Specific losses are used for regression, binary classification or multiclass
classification.
"""
class BaseLoss(ABC):
    """Base class for a loss."""
    def __init__(self, hessians_are_constant) -> None:
        ...
    
    def __call__(self, y_true, raw_predictions, sample_weight):
        """Return the weighted average loss"""
        ...
    
    @abstractmethod
    def pointwise_loss(self, y_true, raw_predictions):
        """Return loss value for each input"""
        ...
    
    need_update_leaves_values = ...
    def init_gradients_and_hessians(self, n_samples, prediction_dim, sample_weight):
        """Return initial gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined
        values.

        Parameters
        ----------
        n_samples : int
            The number of samples passed to `fit()`.

        prediction_dim : int
            The dimension of a raw prediction, i.e. the number of trees
            built at each iteration. Equals 1 for regression and binary
            classification, or K where K is the number of classes for
            multiclass classification.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.

        Returns
        -------
        gradients : ndarray, shape (prediction_dim, n_samples)
            The initial gradients. The array is not initialized.
        hessians : ndarray, shape (prediction_dim, n_samples)
            If hessians are constant (e.g. for `LeastSquares` loss, the
            array is initialized to ``1``. Otherwise, the array is allocated
            without being initialized.
        """
        ...
    
    @abstractmethod
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        """Return initial predictions (before the first iteration).

        Parameters
        ----------
        y_train : ndarray, shape (n_samples,)
            The target training values.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.

        prediction_dim : int
            The dimension of one prediction: 1 for binary classification and
            regression, n_classes for multiclass classification.

        Returns
        -------
        baseline_prediction : float or ndarray, shape (1, prediction_dim)
            The baseline prediction.
        """
        ...
    
    @abstractmethod
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        """Update gradients and hessians arrays, inplace.

        The gradients (resp. hessians) are the first (resp. second) order
        derivatives of the loss for each sample with respect to the
        predictions of model, evaluated at iteration ``i - 1``.

        Parameters
        ----------
        gradients : ndarray, shape (prediction_dim, n_samples)
            The gradients (treated as OUT array).

        hessians : ndarray, shape (prediction_dim, n_samples) or \
            (1,)
            The hessians (treated as OUT array).

        y_true : ndarray, shape (n_samples,)
            The true target values or each training sample.

        raw_predictions : ndarray, shape (prediction_dim, n_samples)
            The raw_predictions (i.e. values from the trees) of the tree
            ensemble at iteration ``i - 1``.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.
        """
        ...
    


class LeastSquares(BaseLoss):
    """Least squares loss, for regression.

    For a given sample x_i, least squares loss is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_pred_i)**2

    This actually computes the half least squares loss to simplify
    the computation of the gradients and get a unit hessian (and be consistent
    with what is done in LightGBM).
    """
    def __init__(self, sample_weight) -> None:
        ...
    
    def pointwise_loss(self, y_true, raw_predictions):
        ...
    
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        ...
    
    @staticmethod
    def inverse_link_function(raw_predictions):
        ...
    
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        ...
    


class LeastAbsoluteDeviation(BaseLoss):
    """Least absolute deviation, for regression.

    For a given sample x_i, the loss is defined as::

        loss(x_i) = |y_true_i - raw_pred_i|
    """
    def __init__(self, sample_weight) -> None:
        ...
    
    need_update_leaves_values = ...
    def pointwise_loss(self, y_true, raw_predictions):
        ...
    
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        ...
    
    @staticmethod
    def inverse_link_function(raw_predictions):
        ...
    
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        ...
    
    def update_leaves_values(self, grower, y_true, raw_predictions, sample_weight):
        ...
    


class Poisson(BaseLoss):
    """Poisson deviance loss with log-link, for regression.

    For a given sample x_i, Poisson deviance loss is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_pred_i))
                    - y_true_i + exp(raw_pred_i))

    This actually computes half the Poisson deviance to simplify
    the computation of the gradients.
    """
    def __init__(self, sample_weight) -> None:
        ...
    
    inverse_link_function = ...
    def pointwise_loss(self, y_true, raw_predictions):
        ...
    
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        ...
    
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        ...
    


class BinaryCrossEntropy(BaseLoss):
    """Binary cross-entropy loss, for binary classification.

    For a given sample x_i, the binary cross-entropy loss is defined as the
    negative log-likelihood of the model which can be expressed as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).
    """
    def __init__(self, sample_weight) -> None:
        ...
    
    inverse_link_function = ...
    def pointwise_loss(self, y_true, raw_predictions):
        ...
    
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        ...
    
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        ...
    
    def predict_proba(self, raw_predictions):
        ...
    


class CategoricalCrossEntropy(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the model and generalizes the binary
    cross-entropy to more than 2 classes.
    """
    def __init__(self, sample_weight) -> None:
        ...
    
    def pointwise_loss(self, y_true, raw_predictions):
        ...
    
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        ...
    
    def update_gradients_and_hessians(self, gradients, hessians, y_true, raw_predictions, sample_weight):
        ...
    
    def predict_proba(self, raw_predictions):
        ...
    


_LOSSES = { 'least_squares': LeastSquares,'least_absolute_deviation': LeastAbsoluteDeviation,'binary_crossentropy': BinaryCrossEntropy,'categorical_crossentropy': CategoricalCrossEntropy,'poisson': Poisson }
