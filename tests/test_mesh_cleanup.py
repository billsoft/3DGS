import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from mesh_cleanup import remove_unsupported_islands

class Field:
    def __init__(self,delta,confidence=1):self.delta=delta;self.confidence=confidence
    def sample(self,p):
        return np.full(len(p),self.delta),np.full(len(p),self.confidence),np.ones(len(p))

class CleanupTests(unittest.TestCase):
    def scene(self):
        return np.array([[0,0,0],[10,0,0],[0,10,0],[3,3,1],[3.1,3,1],[3,3.1,1]],float),np.array([[0,1,2],[3,4,5]])
    def test_small_isolated_contradicted_component_removed(self):
        p,t=self.scene();keep,r=remove_unsupported_islands(p,t,[Field(2),Field(2),Field(0)],.1)
        self.assertEqual(keep.tolist(),[True,False]);self.assertEqual(r['removed_faces'],1)
    def test_two_view_supported_or_unknown_or_protected_retained(self):
        p,t=self.scene()
        for fields in [[Field(2),Field(2),Field(0),Field(0)],[Field(2,0),Field(2,0)],[Field(2)]]:
            self.assertTrue(remove_unsupported_islands(p,t,fields,.1)[0].all())
        self.assertTrue(remove_unsupported_islands(p,t,[Field(2),Field(2)],.1,np.array([False,True]))[0].all())
    def test_attached_structure_and_long_thin_wire_retained(self):
        p,t=self.scene();t[1,0]=0
        self.assertTrue(remove_unsupported_islands(p,t,[Field(2),Field(2)],.1)[0].all())
        p,t=self.scene();p[4,0]=4
        self.assertTrue(remove_unsupported_islands(p,t,[Field(2),Field(2)],.1)[0].all())
    def test_scale_equivariance(self):
        p,t=self.scene()
        a=remove_unsupported_islands(p,t,[Field(2),Field(2)],.1)[0]
        b=remove_unsupported_islands(p*3,t,[Field(6),Field(6)],.3)[0]
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
