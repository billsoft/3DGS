import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from sphere_repair import closed_sphere,propose_spheres,validate
from textured_mesh import smooth_mesh,face_probes

class SphereRepairTests(unittest.TestCase):
    def test_closed_manifold_outward_sphere(self):
        center=np.array([2,-3,4]);p,t=closed_sphere(center,1.5)
        edge,count=np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),1),axis=0,return_counts=True)
        self.assertTrue((count==2).all());self.assertEqual(len(p)-len(edge)+len(t),2)
        np.testing.assert_allclose(np.linalg.norm(p-center,axis=1),1.5,atol=5e-7)
        normal=np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]])
        self.assertTrue(((normal*(p[t].mean(1)-center)).sum(1)>0).all())

    def test_missing_cap_sphere_and_similarity_transformed_input(self):
        p,t=closed_sphere([0,0,0],1)
        spacing=float(np.median(np.linalg.norm(p[t[:,1]]-p[t[:,0]],axis=1)))
        p=p[p[:,2]<.6]
        candidates=propose_spheres(p,spacing,400);self.assertEqual(len(candidates),1)
        np.testing.assert_allclose(candidates[0]['center'],0,atol=1e-5);self.assertAlmostEqual(candidates[0]['radius'],1,places=5)
        R=Rotation.from_rotvec([.4,.8,-.5]).as_matrix();center=np.array([10,-5,8]);shift=p@R.T*3+center
        candidates=propose_spheres(shift,spacing*3,400);self.assertEqual(len(candidates),1)
        np.testing.assert_allclose(candidates[0]['center'],center,atol=1e-4);self.assertAlmostEqual(candidates[0]['radius'],3,places=4)

    def test_plane_and_cylinder_are_not_rounded(self):
        x,y=np.meshgrid(np.linspace(-1,1,40),np.linspace(-1,1,40))
        self.assertEqual(propose_spheres(np.c_[x.ravel(),y.ravel(),np.zeros(x.size)],.06,400),[])
        a,z=np.meshgrid(np.linspace(0,2*np.pi,64,endpoint=False),np.linspace(-1,1,32))
        self.assertEqual(propose_spheres(np.c_[np.cos(a).ravel(),np.sin(a).ravel(),z.ravel()],.07,400),[])

    def test_free_space_evidence_rejects_closed_fill(self):
        class FreeSpace:
            def sample(self,p):return np.ones(len(p)),np.ones(len(p)),np.ones(len(p))*3
        C=np.array([[3,0,0],[-3,0,0],[0,3,0]]);Q=np.repeat(np.eye(3)[None],3,axis=0)
        result=validate(dict(center=[0,0,0],radius=1),[FreeSpace()]*3,[np.zeros((1024,2048,3),'u1')]*3,C,Q,.06)
        self.assertFalse(result['accepted']);self.assertEqual(result['surface_depth_coverage'],0)

    def test_verified_surface_is_not_shrunk_by_smoothing(self):
        p,t=closed_sphere([0,0,0],1,8,16)
        out,_=smooth_mesh(p,t,.1,protected=np.ones(len(p),bool));np.testing.assert_array_equal(out,p)

    def test_face_probes_cover_interior_not_only_centroid(self):
        p=np.array([[0,0,0],[3,0,0],[0,3,0]],dtype='f');q=face_probes(p,np.array([[0,1,2]]))
        self.assertEqual(q.shape,(1,4,3));np.testing.assert_allclose(q[0,0],[1,1,0]);np.testing.assert_allclose(q[0,1],[.2,.2,0])

if __name__=='__main__':unittest.main()
