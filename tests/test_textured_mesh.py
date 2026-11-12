"""Numerical tests for visibility and conservative photo compositing."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from textured_mesh import raster,unwrap_triangle_uv,boundary_weights,blend_linear,smooth_mesh
from mesh_regularize import close_small_holes

class MeshTests(unittest.TestCase):
    def test_near_surface_occludes_far_surface(self):
        xy=np.array([[1,1],[8,1],[1,8]]*2,dtype='f')
        z=np.array([4,4,4,2,2,2],dtype='f');tri=np.arange(6,dtype='i').reshape(2,3)
        depth,face,bary=raster(xy,z,tri,10,10)
        self.assertEqual(face[2,2],1);self.assertAlmostEqual(depth[2,2],2)
        self.assertTrue(np.isinf(depth[9,9]));self.assertEqual(face[9,9],-1)
        self.assertTrue(np.all(bary[2,2]>=0));self.assertLessEqual(bary[2,2].sum(),1)

    def test_perspective_correct_depth(self):
        xy=np.array([[0,0],[6,0],[0,6]],dtype='f');z=np.array([1,2,4],dtype='f')
        depth,face,bary=raster(xy,z,np.array([[0,1,2]],dtype='i'),7,7)
        expected=1/(1/3+1/6+1/12)
        self.assertAlmostEqual(depth[2,2],expected,places=5)
        self.assertAlmostEqual(bary[2,2,0],expected/3,places=5)

    def test_erp_seam_does_not_cross_entire_photo(self):
        uv=np.array([[[.98,.5],[.02,.5],[.01,.6]],[[.2,.2],[.3,.2],[.2,.3]]])
        result=unwrap_triangle_uv(uv)
        self.assertLess(np.ptp(result[0,:,0]),.05)
        np.testing.assert_allclose(result[:,:,0]%1,uv[:,:,0])
        np.testing.assert_allclose(result[1],uv[1])

    def test_boundary_weights_support_arbitrary_source_counts(self):
        for n in [2,4,6,9]:
            tri=np.array([[0,i+1,i+2] for i in range(n)],dtype='i')
            w=boundary_weights(tri,np.arange(n),n,n+2)
            np.testing.assert_allclose(w.sum(1),1)
            np.testing.assert_allclose(w[0],np.full(n,1/n))

    def test_color_blend_is_linear_light_and_preserves_core(self):
        base=np.zeros((1,3));other=np.ones((1,3))
        np.testing.assert_allclose(blend_linear(base,other,np.array([0.])),base)
        np.testing.assert_allclose(blend_linear(base,other,np.array([.5])),.735356983,atol=1e-6)

    def test_open_boundaries_stay_fixed(self):
        p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,.05]],dtype='f')
        tri=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        out,report=smooth_mesh(p,tri,.01)
        np.testing.assert_allclose(out[:4],p[:4]);self.assertLessEqual(report['max_displacement'],.010001)
        self.assertEqual(report['invalid_faces'],0)

    def test_outer_triangle_boundary_is_not_a_hole(self):
        p=np.array([[0,0,0],[.1,0,0],[0,.1,0]],dtype='f')
        added,count=close_small_holes(p,np.array([[0,1,2]]),.1)
        self.assertEqual(count,0);self.assertEqual(len(added),0)

    def test_small_planar_hole_closes_but_large_gap_remains(self):
        p=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[-.05,-.05,0],[.05,-.05,0],[.05,.05,0],[-.05,.05,0]],dtype='f')
        tri=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        added,count=close_small_holes(p,tri,.1)
        self.assertEqual(count,1);self.assertEqual(len(added),2)
        self.assertTrue(np.all(added>=4))

if __name__=='__main__':unittest.main()
