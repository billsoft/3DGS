import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from point_geometry_smooth import smooth_oriented_points

class PointGeometryTests(unittest.TestCase):
    def test_plane_noise_and_motion_bound(self):
        rng=np.random.default_rng(11);x,y=np.meshgrid(np.linspace(-1,1,50),np.linspace(-1,1,50));p=np.c_[x.ravel(),y.ravel(),rng.normal(0,.012,x.size)];n=np.tile([0.,0.,1.],(len(p),1))
        q,nn,r=smooth_oriented_points(p,n,.04,4)
        self.assertLess(q[:,2].std(),p[:,2].std()*.4);self.assertLessEqual(r['max_displacement'],.040001)
        np.testing.assert_allclose(q[:,:2],p[:,:2]);np.testing.assert_allclose(np.linalg.norm(nn,axis=1),1)
    def test_nearby_separate_surfaces_do_not_merge(self):
        rng=np.random.default_rng(4);p=rng.random((800,3));p[:400,2]=0;p[400:,2]=.2;n=np.tile([0.,0.,1.],(800,1))
        q,_,_=smooth_oriented_points(p,n,.02,4);np.testing.assert_allclose(q[:,2],p[:,2],atol=1e-8)

if __name__=='__main__':unittest.main()
