import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from bilateral_appearance import filter_colors

class BilateralTests(unittest.TestCase):
    def plane(self):
        x,y=np.meshgrid(np.arange(20)*.02,np.arange(20)*.02)
        b=np.zeros((400,14),dtype='f');b[:,0]=x.ravel();b[:,1]=y.ravel();b[:,7:10]=np.log([.03,.03,.002]);b[:,10]=1
        s=np.zeros((400,9,3),dtype='f');return b,s
    def test_constant_and_geometry_preserved(self):
        b,s=self.plane();s[:,0]=.2;b[:,3:6]=s[:,0];r,t,_=filter_colors(b,s)
        np.testing.assert_array_equal(r,b);np.testing.assert_array_equal(t,s)
    def test_denoises_without_crossing_color_step(self):
        b,s=self.plane();rng=np.random.default_rng(2);rgb=np.where(b[:,0,None]<.2,.2,.8)+rng.normal(0,.008,(400,3))
        s[:,0]=(rgb-.5)/.2820947918;b[:,3:6]=s[:,0]
        r,t,_=filter_colors(b,s,denoise=.5,sharpen=0);out=t[:,0]*.2820947918+.5
        expected=np.where(b[:,0,None]<.2,.2,.8)
        self.assertLess(np.mean((out-expected)**2),np.mean((rgb-expected)**2))
        np.testing.assert_array_equal(r[:,6:],b[:,6:]);np.testing.assert_array_equal(r[:,:3],b[:,:3]);np.testing.assert_array_equal(t[:,1:],s[:,1:])
        self.assertLess(out[b[:,0]<.2].max(),.25);self.assertGreater(out[b[:,0]>=.2].min(),.75)
    def test_separate_layers_and_isotropic_points_do_not_mix(self):
        b,s=self.plane();upper=b.copy();upper[:,2]=.009
        b=np.concatenate([b,upper]);s=np.concatenate([s,s]);s[400:,0]=.05;b[:,3:6]=s[:,0]
        r,t,_=filter_colors(b,s)
        np.testing.assert_allclose(t,s,atol=1e-6)
        b[:,7:10]=np.log(.02);s[:,0]=np.random.default_rng(4).normal(0,.1,(800,3));b[:,3:6]=s[:,0]
        r,t,_=filter_colors(b,s);np.testing.assert_array_equal(t,s)

if __name__=='__main__':unittest.main()
