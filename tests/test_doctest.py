import doctest
import unittest

import dendritic_modeling.models as models
import dendritic_modeling.dendrinet as dendrinet



def test_doctest_suit():
    test_suit = unittest.TestSuite()

    test_suit.addTest(doctest.DocTestSuite(models))
    test_suit.addTest(doctest.DocTestSuite(dendrinet))


    runner = unittest.TextTestRunner(verbosity=2).run(test_suit)


    assert not runner.failures