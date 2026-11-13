import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from planar_guide_refine import regularize

class PlaneGuideTests(unittest.TestCase):
    def test_planar_prior_requires_three_station_support(self):
        x,y=np.meshgrid(np.linspace(-1,1,30),np.linspace(-1,1,30));p=np.c_[x.ravel(),y.ravel(),np.full(x.size,.01)];n=np.tile([0.,0.,1.],(len(p),1))
        plane=dict(normal=[0,0,1],offset=0,supporting_stations=['a','b'])
        q,nn,r=regularize(p,n,[plane],.08);np.testing.assert_allclose(q,p);self.assertEqual(r['selected_points'],0)
        plane['supporting_stations'].append('c');q,nn,r=regularize(p,n,[plane],.08)
        self.assertGreater(r['selected_points'],800);self.assertLess(abs(q[:,2]).mean(),.002)
        np.testing.assert_allclose(np.linalg.norm(nn,axis=1),1)
    def test_separate_surface_outside_band_is_unchanged(self):
        x,y=np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20));p=np.c_[x.ravel(),y.ravel(),np.full(x.size,.5)];n=np.tile([0.,0.,1.],(len(p),1))
        plane=dict(normal=[0,0,1],offset=0,supporting_stations=['a','b','c'])
        q,nn,r=regularize(p,n,[plane],.08);np.testing.assert_allclose(q,p)
if __name__=='__main__':unittest.main()
