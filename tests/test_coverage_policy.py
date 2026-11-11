import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_next'))
from coverage_policy import replacement_gate,assert_coverage_unchanged,photometric_accept

class CoverageTests(unittest.TestCase):
    def test_better_score_does_not_override_bad_depth_or_visibility(self):
        accepted=replacement_gate(np.ones(3),np.ones(3)*2,np.zeros(3),np.zeros(3),np.array([0,1,0]),np.ones(3),np.array([1,1,.1]))
        np.testing.assert_array_equal(accepted,[True,False,False])
    def test_color_change_preserves_coverage_but_opacity_change_fails(self):
        a=np.zeros((3,14));b=a.copy();b[:,3:6]=1
        assert_coverage_unchanged(a,b)
        b[0,6]=.1
        with self.assertRaises(ValueError):assert_coverage_unchanged(a,b)
    def test_deleting_a_point_fails(self):
        with self.assertRaises(ValueError):assert_coverage_unchanged(np.zeros((3,14)),np.zeros((2,14)))
    def test_average_gain_cannot_hide_a_bad_view_or_single_view_support(self):
        delta=np.array([[-.03,.01],[-.01,-.01],[-.02,0.]])
        valid=np.array([[True,True],[True,True],[True,False]])
        keep,_,_=photometric_accept(delta,valid)
        np.testing.assert_array_equal(keep,[False,True,False])

if __name__=='__main__':unittest.main()
