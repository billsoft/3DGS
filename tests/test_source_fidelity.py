import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from source_fidelity import local_fidelity,strict_source_gate

class SourceFidelityTests(unittest.TestCase):
    def test_no_evidence_is_not_a_pass(self):
        self.assertFalse(strict_source_gate([])['passed'])
    def test_local_damage_not_hidden_by_good_surroundings(self):
        photo=np.full((256,256,3),.5,'f');native=photo.copy();bad=photo.copy();bad[:32,:32]+=.1
        result=local_fidelity(native,bad,photo,np.ones((256,256),bool),32)
        self.assertGreater(result['regressed_tiles'],0)
    def test_blotches_distinguished_from_small_alternating_noise(self):
        photo=np.full((256,256,3),.5,'f');native=photo.copy();native[::2]+=.02;native[1::2]-=.02
        candidate=photo.copy();candidate[:128]+=.02;candidate[128:]-=.02
        result=local_fidelity(native,candidate,photo,np.ones((256,256),bool),32)
        self.assertGreater(result['candidate_blotch'],result['native_blotch']+.01)
    def test_one_bad_direction_fails_whole_source_gate(self):
        good=dict(view='good',native=dict(ssim=.9,mae=.02),fused=dict(ssim=.91,mae=.019),local_fidelity=dict(native_blotch=.01,candidate_blotch=.009,regressed_tiles=0))
        bad=dict(good,view='bad',fused=dict(ssim=.85,mae=.03))
        self.assertTrue(strict_source_gate([good])['passed']);self.assertFalse(strict_source_gate([good,bad])['passed'])

if __name__=='__main__':unittest.main()
