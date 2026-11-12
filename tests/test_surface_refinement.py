import unittest,tempfile,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from surface_refine import initialize
from tour_refine import split_arrays
from oblique_views import make_views
from planar_depth import consolidate


class SurfaceMathTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(12);b=np.zeros((20,14),'f');b[:,:3]=rng.normal(size=(20,3));b[:,7:10]=rng.uniform(-4,-1,(20,3))
        b[:,10:14]=Rotation.random(20,random_state=rng).as_quat()[:,[3,0,1,2]];s=rng.normal(size=(20,9,3)).astype('f');b[:,3:6]=s[:,0]
        return b,s

    def test_surface_axes_preserve_covariance(self):
        b,s=self.sample()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.npz';np.savez(path,base=b,sh=s);q,t=initialize(path)
        def covariance(x):
            r=Rotation.from_quat(x[:,[11,12,13,10]]).as_matrix()
            return (r*np.exp(2*x[:,None,7:10]))@r.transpose(0,2,1)
        np.testing.assert_allclose(covariance(b),covariance(q),rtol=2e-6,atol=1e-8)
        self.assertTrue(np.all(q[:,9]<=q[:,7]));self.assertTrue(np.all(q[:,9]<=q[:,8]));np.testing.assert_array_equal(t,s)

    def test_split_conserves_center_and_appearance(self):
        b,s=self.sample();ids=np.array([2,5,9]);q,t=split_arrays(b,s,ids)
        self.assertEqual(len(q),23);np.testing.assert_allclose(q[-6:,:3].reshape(3,2,3).mean(1),b[ids,:3],atol=1e-7)
        np.testing.assert_array_equal(t[-6:],np.repeat(s[ids],2,axis=0));self.assertTrue(np.isfinite(q).all())

    def test_arbitrary_camera_count_and_oblique_rays(self):
        for n in [1,2,6,9]:
            c=np.c_[np.arange(n),np.zeros((n,2))];q=np.repeat(np.eye(3)[None],n,axis=0)
            views=make_views(c,q,[(i,i+1) for i in range(n-1)])
            self.assertEqual(len(views),6*n+10*(n-1))
            for v in views:
                V=np.array(v['viewmat']);B=np.array(v['basis']);C=np.array(v['center'])
                np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-7);self.assertGreater(np.linalg.det(B),.999)
                np.testing.assert_allclose(V@np.r_[C,1],[0,0,0,1],atol=1e-7)
                np.testing.assert_allclose(V@np.r_[C+B[:,2],1],[0,0,1,1],atol=1e-7)

    def test_plane_cleanup_preserves_foreground_and_missing_depth(self):
        y,x=np.mgrid[:64,:64];depth=(2+.002*np.sin(x/5)).astype('f');alpha=np.ones_like(depth)
        depth[20:30,20:30]=1.;depth[40:42,40:42]=0;alpha[40:42,40:42]=0
        p=[dict(normal=[0,0,1],offset=-2.)]
        result,owner=consolidate(depth,alpha,np.eye(4),p,.04)
        np.testing.assert_array_equal(result[20:30,20:30],depth[20:30,20:30])
        np.testing.assert_array_equal(result[40:42,40:42],depth[40:42,40:42])
        self.assertGreater((owner>=0).sum(),3000)
        np.testing.assert_allclose(result[2:15,2:15],2.,atol=1e-7)

    def test_vertical_camera_baseline_uses_finite_horizontal_look(self):
        poses=make_views([[0,0,0],[0,0,1]],np.repeat(np.eye(3)[None],2,axis=0),[(0,1)])
        self.assertTrue(all(np.isfinite(p['viewmat']).all() for p in poses))


if __name__=='__main__':unittest.main()
