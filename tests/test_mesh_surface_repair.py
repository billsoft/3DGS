import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
try:
    from mesh_surface_repair import repair_arrays,visibility
except ImportError:
    repair_arrays=visibility=None

@unittest.skipIf(repair_arrays is None,'Requires isolated Open3D environment')
class SurfaceRepairTests(unittest.TestCase):
    def test_cleanup_preserves_long_thin_strip(self):
        # Tiny floating triangle removed; long thin strip must survive size gate.
        p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[1,0,0],[2,0,0],[2,.002,0],[1,.002,0]],'f')
        tri=np.array([[0,1,2],[3,4,5],[3,5,6]],'i')
        q,t,cap,report=repair_arrays(p,tri,.02,24,3)
        self.assertEqual(report['removed_faces'],1)
        self.assertEqual(len(t),2);np.testing.assert_array_equal(q,p)

    def test_small_hole_filled_outer_boundary_retained(self):
        x,y=np.meshgrid(np.arange(7)*.1,np.arange(7)*.1)
        p=np.c_[x.ravel(),y.ravel(),np.zeros(49)].astype('f');tri=[]
        for j in range(6):
            for i in range(6):
                if (i,j)==(3,3):continue
                a=j*7+i;tri.extend([[a,a+1,a+8],[a,a+8,a+7]])
        tri=np.array(tri,'i');q,t,cap,r=repair_arrays(p,tri,.1,0,1)
        self.assertEqual(r['added_faces'],2);self.assertEqual(len(t),72)
        np.testing.assert_array_equal(q,p)

    def test_raycast_rejects_occluded_surface(self):
        p=np.array([[-1,-1,2],[1,-1,2],[0,1,2],[-1,-1,4],[1,-1,4],[0,1,4]],'f')
        visible,coverage=visibility(p,np.array([[0,1,2],[3,4,5]],'i'),np.array([[0,0,0]]),.01)
        self.assertTrue(visible[0,0]);self.assertFalse(visible[1,0])
        self.assertEqual(coverage[0,0],1);self.assertEqual(coverage[1,0],0)

    def test_larger_nonplanar_hole_is_not_flattened(self):
        x,y=np.meshgrid(np.arange(7)*.1,np.arange(7)*.1)
        p=np.c_[x.ravel(),y.ravel(),np.zeros(49)].astype('f');tri=[]
        for j in range(6):
            for i in range(6):
                if (i,j)==(3,3):continue
                a=j*7+i;tri.extend([[a,a+1,a+8],[a,a+8,a+7]])
        p[3*7+3,2]=.06
        q,t,cap,r=repair_arrays(p,np.array(tri,'i'),.02,0,8)
        self.assertGreater(r['rejected_large_curved_caps'],0)
        self.assertFalse(cap.any());np.testing.assert_array_equal(q,p)

    def test_partial_occlusion_keeps_visible_triangle_corners(self):
        p=np.array([[-.2,-.2,2],[.2,-.2,2],[0,.2,2],[-2,-1,4],[2,-1,4],[0,2,4]],'f')
        seen,coverage=visibility(p,np.array([[0,1,2],[3,4,5]],'i'),np.array([[0,0,0]]),.01)
        self.assertTrue(seen[1,0]);self.assertEqual(coverage[1,0],.75)

if __name__=='__main__':unittest.main()
