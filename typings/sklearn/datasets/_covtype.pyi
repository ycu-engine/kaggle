"""
This type stub file was generated by pyright.
"""

import logging
from ._base import RemoteFileMetadata
from ..utils.validation import _deprecate_positional_args

"""Forest covertype dataset.

A classic dataset for classification benchmarks, featuring categorical and
real-valued features.

The dataset page is available from UCI Machine Learning Repository

    https://archive.ics.uci.edu/ml/datasets/Covertype

Courtesy of Jock A. Blackard and Colorado State University.
"""
ARCHIVE = RemoteFileMetadata(filename='covtype.data.gz', url='https://ndownloader.figshare.com/files/5976039', checksum='614360d0257557dd1792834a85a1cdeb' 'fadc3c4f30b011d56afee7ffb5b15771')
logger = logging.getLogger(__name__)
FEATURE_NAMES = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
TARGET_NAMES = ["Cover_Type"]
@_deprecate_positional_args
def fetch_covtype(*, data_home=..., download_if_missing=..., random_state=..., shuffle=..., return_X_y=..., as_frame=...):
    """Load the covertype dataset (classification).

    Download it if necessary.

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

    Read more in the :ref:`User Guide <covtype_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas DataFrame or
        Series depending on the number of target columns. If `return_X_y` is
        True, then (`data`, `target`) will be pandas DataFrames or Series as
        described below.

        .. versionadded:: 0.24

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (581012, 54)
            Each row corresponds to the 54 features in the dataset.
        target : ndarray of shape (581012,)
            Each value corresponds to one of
            the 7 forest covertypes with values
            ranging between 1 to 7.
        frame : dataframe of shape (581012, 53)
            Only present when `as_frame=True`. Contains `data` and `target`.
        DESCR : str
            Description of the forest covertype dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    """
    ...
