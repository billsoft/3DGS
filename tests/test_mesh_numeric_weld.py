import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from mesh_numeric_weld import weld

class WeldTests(unittest.TestCase):
    def test_duplicate_vertex_seams_close_without_filling_real_holes(self):
        import open3d as o
        m=o.geometry.TriangleMesh.create_box();p=np.asarray(m.vertices);t=np.asarray(m.triangles)
        duplicated=p[t].reshape(-1,3);faces=np.arange(len(duplicated)).reshape(-1,3)
        q,f,r=weld(duplicated,faces,1e-5)
        self.assertEqual(len(q),8);self.assertEqual(r['boundary_edges'],0);self.assertEqual(r['tiny_faces'],0)
        q,f,r=weld(duplicated,faces[:-1],1e-5);self.assertEqual(r['boundary_edges'],3)
    def test_distinct_nearby_surfaces_remain_separate(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.01],[1,0,.01],[0,1,.01]],float)
        q,f,r=weld(p,np.array([[0,1,2],[3,4,5]]),1e-5)
        self.assertEqual(len(q),6);self.assertEqual(len(f),2)
if __name__=='__main__':unittest.main()
