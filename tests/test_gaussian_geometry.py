import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_next'))
from gaussian_geometry import shortest_axis_normals

class NormalTests(unittest.TestCase):
    def test_equivalent_axis_permutations_preserve_normal(self):
        a=np.zeros((2,14));a[:,10]=1;a[0,7:10]=[-3,-1,0]
        frame=np.eye(3)[:,[1,2,0]]
        a[1,7:10]=[-1,0,-3];q=Rotation.from_matrix(frame).as_quat();a[1,10:14]=q[[3,0,1,2]]
        normals=shortest_axis_normals(a)
        self.assertAlmostEqual(abs(float(normals[0]@normals[1])),1.)

if __name__=='__main__':unittest.main()
