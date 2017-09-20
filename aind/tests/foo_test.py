#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A starting template for writing unit tests to run and debug "unit" of codes,
as classes and functions. A folder of tests should be placed at each subfolder
of the source code repository. The test should be run from the top-level
directory of this folder calling "python -m tests.foo_test"

Source: Beginning test-driven development...(https://goo.gl/LwQNc4)

@author: ucaiado

Created on MM/DD/YYYY
"""

import unittest
import foo

'''
Begin help functions
'''


'''
End help functions
'''


class TestFoo(unittest.TestCase):

    def test_calculator_add_method_returns_correct_result(self):
        calc = foo.Calculator()
        result = calc.add(2, 2)
        self.assertEqual(4, result)


if __name__ == '__main__':
    unittest.main()
