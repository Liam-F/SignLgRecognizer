from unittest import TestCase

from aind.asl_data import AslDb
from aind.my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)

FEATURES = ['right-y', 'right-x']


class TestSelectors(TestCase):
    def setUp(self):
        asl = AslDb()
        self.training = asl.build_training(FEATURES)
        self.sequences = self.training.get_all_sequences()
        self.xlengths = self.training.get_all_Xlengths()

    def test_select_constant_interface(self):
        xlengths = self.xlengths
        model = SelectorConstant(self.sequences, xlengths, 'BUY').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorConstant(self.sequences, xlengths, 'BOOK').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_bic_interface(self):
        xlengths = self.xlengths
        model = SelectorBIC(self.sequences, xlengths, 'FRANK').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorBIC(self.sequences, xlengths, 'VEGETABLE').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_cv_interface(self):
        xlengths = self.xlengths
        model = SelectorCV(self.sequences, xlengths, 'JOHN').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorCV(self.sequences, xlengths, 'CHICKEN').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_dic_interface(self):
        xlengths = self.xlengths
        model = SelectorDIC(self.sequences, xlengths, 'MARY').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorDIC(self.sequences, xlengths, 'TOY').select()
        self.assertGreaterEqual(model.n_components, 2)
