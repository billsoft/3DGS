import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from photo_recolor import sample_erp,recolor_weight,footprint_span

class PhotoRecolorTests(unittest.TestCase):
    def test_erp_poles_clamp_instead_of_wrapping_to_opposite_pole(self):
        image=np.repeat(np.arange(8,dtype='f')[:,None],16,axis=1)
        values=sample_erp(image,np.array([[0,0,1],[0,0,-1]],float),np.zeros(3),np.eye(3))
        np.testing.assert_allclose(values,[0,7])
    def test_projection_respects_camera_pose(self):
        image=np.arange(128,dtype='f').reshape(8,16);p=np.array([[1,2,3],[-2,1,4]],float)
        angle=.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        center=np.array([3.,2.,1.]);a=sample_erp(image,p,np.zeros(3),np.eye(3));b=sample_erp(image,p@R.T+center,center,R)
        np.testing.assert_allclose(a,b)
    def test_occlusion_and_texture_reject_unsafe_recoloring(self):
        weights=recolor_weight(np.array([0.,1.,0.,0.]),np.array([1.,1.,.1,1.]),np.ones(4),np.array([0.,0.,0.,.2]))
        np.testing.assert_array_equal(weights,[1,0,0,0])
    def test_footprint_detects_edge_that_center_misses(self):
        image=np.zeros((64,128,3),'u1');image[:,64:]=255
        a=np.zeros((2,14),'f');a[:,:3]=[1.,.15,0.];a[:,10]=1;a[:,7:10]=np.log(.001);a[1,7:10]=np.log(.15)
        span=footprint_span(image,a,np.zeros(3),np.eye(3))
        self.assertLess(span[0],.01);self.assertGreater(span[1],.9)

if __name__=='__main__':unittest.main()
