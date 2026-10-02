import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci"))
from retained_row_integrity import verify_rows

class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'id':'a','label':1,'prediction':1,'probabilities':[.1,.7,.1,.1]},
                   {'id':'b','label':0,'prediction':2,'probabilities':[.1,.1,.7,.1]}]
    def verify(self, rows):return verify_rows(rows,['a','b'],[1,0])
    def test_valid(self):self.assertEqual(self.verify(self.rows)['accuracy'],.5)
    def test_labels_must_match_data(self):
        self.rows[0]['label']=2
        with self.assertRaisesRegex(ValueError,'canonical'):self.verify(self.rows)
    def test_order(self):
        with self.assertRaises(ValueError):self.verify(list(reversed(self.rows)))
    def test_missing(self):
        with self.assertRaises(ValueError):self.verify(self.rows[:1])
    def test_empty(self):
        with self.assertRaises(ValueError):verify_rows([],[],[])
    def test_duplicates(self):
        with self.assertRaises(ValueError):verify_rows(self.rows,['a','a'],[1,0])
    def test_nan(self):
        self.rows[0]['probabilities'][0]=float('nan')
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_infinity(self):
        self.rows[0]['probabilities'][0]=float('inf')
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_probability_bounds(self):
        self.rows[0]['probabilities']=[-.1,.9,.1,.1]
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_probability_sum(self):
        self.rows[0]['probabilities']=[.1,.6,.1,.1]
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_argmax(self):
        self.rows[0]['prediction']=2
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_bool_class(self):
        self.rows[0]['prediction']=True
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_float_label(self):
        self.rows[0]['label']=1.0
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_wrong_width(self):
        self.rows[0]['probabilities']=[.1,.9]
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_out_of_range_prediction(self):
        self.rows[0]['prediction']=4
        with self.assertRaises(ValueError):self.verify(self.rows)
    def test_no_mutation(self):
        before=copy.deepcopy(self.rows);self.verify(self.rows);self.assertEqual(before,self.rows)

if __name__=='__main__':unittest.main()
