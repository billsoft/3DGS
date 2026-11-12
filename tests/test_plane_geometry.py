import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from apply_planes import plane_projection
from planar_constraints import triangulate
from resample_planes import plane_samples
from refine_boundaries import refine_quad
import cv2 as cv

class PlaneGeometryTests(unittest.TestCase):
    def test_projection_preserves_rays_and_matches_jacobian(self):
        C=np.array([.1,.2,0]);p=np.array([[.4,.6,3.],[-.5,.4,4.]]);corners=np.array([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],dtype=float);n=np.array([0.,0.,1.])
        q,J,ok,_=plane_projection(p,C,corners,n);self.assertTrue(ok.all());np.testing.assert_allclose(q[:,2],2.)
        np.testing.assert_allclose((q-C)/np.linalg.norm(q-C,axis=1)[:,None],(p-C)/np.linalg.norm(p-C,axis=1)[:,None])
        for k in range(3):
            h=np.eye(3)[k]*1e-5;plus=plane_projection(p+h,C,corners,n)[0];minus=plane_projection(p-h,C,corners,n)[0]
            np.testing.assert_allclose((plus-minus)/(2e-5),J[:,:,k],atol=1e-8)
    def test_foreground_and_outside_polygon_are_not_flattened(self):
        corners=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],float)
        _,_,ok,_=plane_projection(np.array([[0,0,1],[10,0,3],[0,0,3.]]),np.zeros(3),corners,np.array([0.,0.,1.]))
        np.testing.assert_array_equal(ok,[False,False,True])
    def test_triangulation_recovers_physical_points(self):
        C1=np.array([-1.,0,0]);C2=np.array([1.,0,0]);p=np.array([[-.5,-.2,3],[.5,.4,4]])
        r1=p-C1;r1/=np.linalg.norm(r1,axis=1)[:,None];r2=p-C2;r2/=np.linalg.norm(r2,axis=1)[:,None]
        q,gap,t,s,_=triangulate(C1,r1,C2,r2);np.testing.assert_allclose(q,p,atol=1e-12);self.assertTrue((t>0).all() and (s>0).all());np.testing.assert_allclose(gap,0,atol=1e-12)
    def test_surface_sampling_stays_on_bounded_plane(self):
        corners=np.array([[0,0,2],[2,0,2],[2,1,2],[0,1,2]],float)
        p,R=plane_samples(corners,np.array([0.,0.,1.]),.1)
        self.assertEqual(len(p),200);np.testing.assert_allclose(p[:,2],2.)
        self.assertTrue(((p[:,:2]>0)&(p[:,:2]<[2,1])).all());self.assertAlmostEqual(np.linalg.det(R),1.)
    def test_photo_edge_refinement_reduces_known_corner_error(self):
        im=np.full((256,256,3),180,np.uint8);cv.rectangle(im,(40,60),(210,190),(20,20,20),-1)
        truth=np.array([[40,60],[210,60],[210,190],[40,190]],float);guess=truth+np.array([2.,-2.])
        refined,ok=refine_quad(im,guess)
        self.assertTrue(ok);self.assertLess(np.linalg.norm(refined-truth,axis=1).mean(),np.linalg.norm(guess-truth,axis=1).mean())
    def test_featureless_photo_does_not_invent_quad_evidence(self):
        im=np.full((256,256,3),128,np.uint8);q=np.array([[40,60],[210,60],[210,190],[40,190]],float)
        refined,ok=refine_quad(im,q);self.assertFalse(ok);np.testing.assert_array_equal(refined,q)
    def test_projection_is_equivariant_to_scene_units(self):
        corners=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],float);p=np.array([[0,0,1.9],[0,0,2.1],[4,0,3.]])
        q,J,keep,_=plane_projection(p,np.zeros(3),corners,np.array([0.,0.,1.]))
        scaled,Js,ks,_=plane_projection(p*100,np.zeros(3),corners*100,np.array([0.,0.,1.]))
        np.testing.assert_allclose(scaled,q*100);np.testing.assert_allclose(Js,J);np.testing.assert_array_equal(ks,keep)

if __name__=='__main__':unittest.main()
