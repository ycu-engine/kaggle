"""
This type stub file was generated by pyright.
"""

import pytest

def has_verdana():
    """Helper to verify if Verdana font is present"""
    ...

@pytest.fixture(scope="session", autouse=True)
def remove_pandas_unit_conversion():
    ...

@pytest.fixture(autouse=True)
def close_figs():
    ...

@pytest.fixture(autouse=True)
def random_seed():
    ...

@pytest.fixture()
def rng():
    ...

@pytest.fixture
def wide_df(rng):
    ...

@pytest.fixture
def wide_array(wide_df):
    ...

@pytest.fixture
def flat_series(rng):
    ...

@pytest.fixture
def flat_array(flat_series):
    ...

@pytest.fixture
def flat_list(flat_series):
    ...

@pytest.fixture(params=["series", "array", "list"])
def flat_data(rng, request):
    ...

@pytest.fixture
def wide_list_of_series(rng):
    ...

@pytest.fixture
def wide_list_of_arrays(wide_list_of_series):
    ...

@pytest.fixture
def wide_list_of_lists(wide_list_of_series):
    ...

@pytest.fixture
def wide_dict_of_series(wide_list_of_series):
    ...

@pytest.fixture
def wide_dict_of_arrays(wide_list_of_series):
    ...

@pytest.fixture
def wide_dict_of_lists(wide_list_of_series):
    ...

@pytest.fixture
def long_df(rng):
    ...

@pytest.fixture
def long_dict(long_df):
    ...

@pytest.fixture
def repeated_df(rng):
    ...

@pytest.fixture
def missing_df(rng, long_df):
    ...

@pytest.fixture
def object_df(rng, long_df):
    ...

@pytest.fixture
def null_series(flat_series):
    ...

