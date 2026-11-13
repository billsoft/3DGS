import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from depth_geometry_smooth import smooth_depth
from depth_geometry_cloud import frame_points
from depth_source_support import classify_depth

class GeometrySmoothTests(unittest.TestCase):
    def test_occlusion_is_not_free_space_contradiction(self):
        agree,free=classify_depth(np.array([2.,1.,3.,1.]),np.full(4,2.),np.array([True,True,True,False]),.1)
        np.testing.assert_array_equal(agree,[True,False,False,False])
        np.testing.assert_array_equal(free,[False,True,False,False])
    def test_geometry_cloud_pose_and_normal_orientation(self):
        d=np.full((32,32),2.,'f');view=np.eye(4);view[0,3]=-1
        p,n=frame_points(d,np.ones_like(d),view,2)
        self.assertGreater(len(p),100);np.testing.assert_allclose(p[:,2],2)
        self.assertTrue((n[:,2]<-.99).all());self.assertTrue(np.isfinite(p).all())
    def test_reduces_plane_noise_without_depth_bias(self):
        rng=np.random.default_rng(81);depth=(2+rng.normal(0,.009,(96,96))).astype('f')
        result,r=smooth_depth(depth,np.ones_like(depth),'strong')
        self.assertLess(result[12:-12,12:-12].std(),depth[12:-12,12:-12].std()*.3)
        self.assertLess(abs(result[12:-12,12:-12].mean()-2),.002)
        self.assertLess(r['roughness_after'],r['roughness_before']*.3)
    def test_foreground_and_thin_strip_are_preserved(self):
        depth=np.full((96,96),4.,'f');depth[:,:48]=2;depth[:,72:75]=1
        result,_=smooth_depth(depth,np.ones_like(depth))
        np.testing.assert_allclose(result,depth,atol=1e-5)
    def test_invalid_stays_invalid_and_alpha_not_modified(self):
        depth=np.full((96,96),2.,'f');depth[40:45,40:45]=np.nan
        alpha=np.ones_like(depth);alpha[60:65,60:65]=.1;before=alpha.copy()
        result,_=smooth_depth(depth,alpha)
        self.assertTrue(np.isnan(result[40:45,40:45]).all());np.testing.assert_array_equal(alpha,before)
        np.testing.assert_array_equal(result[60:65,60:65],depth[60:65,60:65])
    def test_slanted_plane_not_flattened(self):
        y,x=np.mgrid[:96,:96];depth=(1/(.4+.001*x+.0005*y)).astype('f')
        result,_=smooth_depth(depth,np.ones_like(depth))
        self.assertLess(np.max(abs(result[15:-15,15:-15]-depth[15:-15,15:-15])),.001)

if __name__=='__main__':unittest.main()
