import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_next'))
from territory import domains,ray_plane_jacobian

class TerritoryTests(unittest.TestCase):
    def test_visible_camera_core_cannot_be_overridden(self):
        centers=np.c_[np.linspace(-2,2,41),np.zeros(41),np.zeros(41)]
        cameras=np.array([[-2,0,1],[2,0,1]])
        scores=np.ones((41,2));a=np.arange(40);b=a+1
        macro,distance,allowed,_=domains(centers,cameras,scores,a,b,band=.3)
        self.assertEqual(macro[0],0);self.assertEqual(macro[-1],1)
        self.assertEqual(allowed[0].sum(),1);self.assertEqual(allowed[-1].sum(),1)
        self.assertTrue((allowed[distance>=.3].sum(1)==1).all())

    def test_invisible_nearest_camera_cannot_claim_surface(self):
        centers=np.array([[0,0,0],[.1,0,0]])
        cameras=np.array([[0,0,1],[10,0,1]])
        scores=np.array([[0.,1],[0,1]])
        macro,_,allowed,_=domains(centers,cameras,scores,np.array([0]),np.array([1]))
        np.testing.assert_array_equal(macro,[1,1]);self.assertFalse(allowed[:,0].any())

    def test_quality_can_override_distance_without_forcing_invisible_data(self):
        centers=np.array([[0.,0,0],[.1,0,0]])
        cameras=np.array([[-1.,0,1],[2,0,1]])
        scores=np.array([[.1,1.],[.1,1.]])
        macro,_,_,_=domains(centers,cameras,scores,np.array([0]),np.array([1]))
        np.testing.assert_array_equal(macro,[1,1])

    def test_unobserved_surface_stays_unknown(self):
        macro,_,allowed,_=domains(np.array([[0.,0,0],[1,0,0]]),np.array([[0.,0,1],[2,0,1]]),np.zeros((2,2)),np.array([0]),np.array([1]))
        np.testing.assert_array_equal(macro,[4,4]);self.assertFalse(allowed.any())

    def test_ray_projection_preserves_source_screen_differential(self):
        point=np.array([.2,.3,2.]);ray=(point/np.linalg.norm(point))[None]
        ratio=np.array([1.1]);normal=np.array([[0.,0,1]])
        J=ray_plane_jacobian(ray,normal,ratio)[0]
        def screen(p):
            x,y,z=p;return np.array([[1/z,0,-x/z**2],[0,1/z,-y/z**2]])
        np.testing.assert_allclose(screen(point*ratio[0])@J,screen(point),atol=1e-12)
        np.testing.assert_allclose(J@ray[0],0,atol=1e-12)

if __name__=='__main__':unittest.main()
