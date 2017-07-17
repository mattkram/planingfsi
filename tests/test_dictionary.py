from planingfsi import io
import math
import unittest


class DictionaryTest(unittest.TestCase):

    def setUp(self):
        self.dict_ = io.Dictionary('testDict')
#        for key, val in self.dict_.items():
#            print('{0}: {1}'.format(key, val), type(val))
#            if isinstance(val, dict):
#                for keyj, valj in val.items():
#                    print('{0} = {1}'.format(keyj, valj), type(valj))

    def assertEqualInstance(self, key, target_val, type_):
        dict_val = self.dict_.read(key)
        self.assertIsInstance(dict_val, type_)
        self.assertEqual(target_val, dict_val)
    
    def test_read_bool(self):
        self.assertEqualInstance('boolFalseUpper', False, bool)
        self.assertEqualInstance('boolFalseLower', False, bool)
        self.assertEqualInstance('boolTrueUpper', True, bool)
        self.assertEqualInstance('boolTrueLower', True, bool)

    def test_read_int(self):
        self.assertEqualInstance('intSingleDigit', 2, int)
        self.assertEqualInstance('intMultipleDigits', 2999, int)

    def test_read_nan(self):
        for key in ['nanUnsigned', 'nanNegative', 'nanPositive']:
            val = self.dict_.read(key)
            self.assertIsInstance(val, float)
            self.assertTrue(math.isnan(val))

    def test_read_infs(self):
        INF = float('inf')
        self.assertEqualInstance('infUnsigned', INF, float)
        self.assertEqualInstance('infPositive', INF, float)
        self.assertEqualInstance('infNegative', -INF, float)
    
    def test_read_none(self):
        self.assertIsNone(self.dict_.read('noneUpper'))
        self.assertIsNone(self.dict_.read('noneLower'))

if __name__ == '__main__':
    unittest.main()

