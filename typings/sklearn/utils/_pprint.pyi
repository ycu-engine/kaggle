"""
This type stub file was generated by pyright.
"""

import pprint

"""This module contains the _EstimatorPrettyPrinter class used in
BaseEstimator.__repr__ for pretty-printing estimators"""
class KeyValTuple(tuple):
    """Dummy class for correctly rendering key-value tuples from dicts."""
    def __repr__(self):
        ...
    


class KeyValTupleParam(KeyValTuple):
    """Dummy class for correctly rendering key-value tuples from parameters."""
    ...


class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object reccursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).

    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """
    def __init__(self, indent=..., width=..., depth=..., stream=..., *, compact=..., indent_at_name=..., n_max_elements_to_show=...) -> None:
        ...
    
    def format(self, object, context, maxlevels, level):
        ...
    
    _dispatch = ...


