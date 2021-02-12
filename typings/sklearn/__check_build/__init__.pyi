"""
This type stub file was generated by pyright.
"""

import os

""" Module to give helpful messages to the user that did not
compile scikit-learn properly.
"""
INPLACE_MSG = """
It appears that you are importing a local scikit-learn source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""
STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""
def raise_build_error(e):
    ...
