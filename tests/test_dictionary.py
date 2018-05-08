import math
import os.path
import unittest

import context

from planingfsi import io


DEBUG = False


class DictionaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dict_ = io.Dictionary(os.path.join(context.TEST_DIR, 'testDict'))
        if DEBUG:
            for key, val in cls.dict_.items():
                print('{0}: {1}'.format(key, val), type(val))
                if isinstance(val, dict):
                    for keyj, valj in val.items():
                        print('{0} = {1}'.format(keyj, valj), type(valj))

    def assertEqualInstance(self, key, target_val, type_):
        dict_val = self.dict_.read(key)
        self.assertIsInstance(dict_val, type_)
        self.assertEqual(target_val, dict_val)

    def assertDict(self, dict_, type_):
        for key, val in dict_.items():
            with self.subTest(key=key):
                self.assertEqualInstance(key, val, type_)

    def test_read_bool(self):
        dict_ = {'boolTrueUpper': True,
                 'boolTrueLower': True,
                 'boolFalseUpper': False,
                 'boolFalseLower': False}
        self.assertDict(dict_, bool)

    def test_read_int(self):
        dict_ = {'intMultipleDigits': 2999,
                 'intSingleDigit': 2}
        self.assertDict(dict_, int)

    def test_read_nan(self):
        for key in ['nanUnsigned', 'nanNegative', 'nanPositive']:
            val = self.dict_.read(key)
            self.assertIsInstance(val, float)
            self.assertTrue(math.isnan(val))

    def test_read_infs(self):
        INF = float('inf')
        dict_ = {'infUnsigned': INF,
                 'infPositive': INF,
                 'infNegative': -INF}
        self.assertDict(dict_, float)

    def test_read_none(self):
        for upper_lower in ['Upper', 'Lower']:
            self.assertIsNone(self.dict_.read('none{0}'.format(upper_lower)))

if __name__ == '__main__':
    unittest.main()

