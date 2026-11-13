import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from local_quality import local_metrics

class LocalQualityTests(unittest.TestCase):
    def test_isolates_small_color_halo_despite_small_whole_image_error(self):
        target=np.full((256,256,3),.5);a=target.copy();a[64:96,64:96]+=[-.15,.15,0]
        result=local_metrics(a,target,np.ones((256,256)))
        self.assertGreater(result['patch_max'],.09)
        self.assertLess(abs(a-target).mean(),.002)
        self.assertGreater(result['chroma_mae'],0)
    def test_masked_holes_do_not_become_bad_color(self):
        target=np.ones((64,64,3));a=target.copy();a[:32]=0;alpha=np.ones((64,64));alpha[:32]=0
        result=local_metrics(a,target,alpha)
        self.assertEqual(result['patch_max'],0);self.assertEqual(result['valid_patches'],2)
    def test_empty_support_is_explicit(self):
        a=np.zeros((32,32,3));result=local_metrics(a,a,np.zeros((32,32)))
        self.assertIsNone(result['chroma_mae']);self.assertIsNone(result['patch_p95'])

if __name__=='__main__':unittest.main()
