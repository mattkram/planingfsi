import math

from planingfsi.io import Dictionary


def assert_equal_instance(dict_, key, target_val, type_):
    """Assert the dictionary value equals the target and is of the expected type."""
    dict_val = dict_.read(key)
    assert isinstance(dict_val, type_)
    assert target_val == dict_val


def assert_dict(dict_, type_):
    """Assert that all instances in a dict have values of the required type."""
    for key, val in dict_.items():
        assert_equal_instance(dict_, key, val, type_)


def test_read_bool():
    dict_ = Dictionary({
        'boolTrueUpper': True,
        'boolTrueLower': True,
        'boolFalseUpper': False,
        'boolFalseLower': False
    })
    assert_dict(dict_, bool)


def test_read_int():
    dict_ = Dictionary({
        'intMultipleDigits': 2999,
        'intSingleDigit': 2
    })
    assert_dict(dict_, int)


def test_read_nan(test_dict):
    for key in ['nanUnsigned', 'nanNegative', 'nanPositive']:
        val = test_dict.read(key)
        assert isinstance(val, float)
        assert math.isnan(val)


def test_read_infs():
    infinity = float('inf')
    dict_ = Dictionary({
        'infUnsigned': infinity,
        'infPositive': infinity,
        'infNegative': -infinity
    })
    assert_dict(dict_, float)


def test_read_none(test_dict):
    for upper_lower in ['Upper', 'Lower']:
        assert test_dict.read('none{0}'.format(upper_lower)) is None


def test_read_env_var(test_dict):
    assert test_dict['envVar'] == 'Dummy'
