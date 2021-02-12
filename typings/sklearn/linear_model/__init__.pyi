"""
This type stub file was generated by pyright.
"""

from ._base import LinearRegression
from ._bayes import ARDRegression, BayesianRidge
from ._least_angle import Lars, LarsCV, LassoLars, LassoLarsCV, LassoLarsIC, lars_path, lars_path_gram
from ._coordinate_descent import ElasticNet, ElasticNetCV, Lasso, LassoCV, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV, enet_path, lasso_path
from ._glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from ._huber import HuberRegressor
from ._sgd_fast import Hinge, Huber, Log, ModifiedHuber, SquaredLoss
from ._stochastic_gradient import SGDClassifier, SGDRegressor
from ._ridge import Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, ridge_regression
from ._logistic import LogisticRegression, LogisticRegressionCV
from ._omp import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, orthogonal_mp, orthogonal_mp_gram
from ._passive_aggressive import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from ._perceptron import Perceptron
from ._ransac import RANSACRegressor
from ._theil_sen import TheilSenRegressor

"""
The :mod:`sklearn.linear_model` module implements a variety of linear models.
"""