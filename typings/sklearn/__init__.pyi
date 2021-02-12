"""
This type stub file was generated by pyright.
"""

import sys
import logging
import os
import random
from ._config import config_context, get_config, set_config
from . import __check_build, _distributor_init
from .base import clone
from .utils._show_versions import show_versions

"""
Machine learning module for Python
==================================

sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.org for complete documentation.
"""
logger = logging.getLogger(__name__)
__version__ = '0.24.1'
if __SKLEARN_SETUP__:
    ...
else:
    ...
def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    ...
