import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from sh_io import write_sh_ply,read_sh_ply
from input_io import header,read_gaussians

class SHIOTests(unittest.TestCase):
    def test_channel_major_standard_ply_and_exact_roundtrip(self):
        base=np.zeros((2,14),'f');base[:,10]=1;base[:,:3]=[[1,2,3],[4,5,6]]
        sh=np.arange(54,dtype='f').reshape(2,9,3)/100;base[:,3:6]=sh[:,0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.ply';write_sh_ply(p,base,sh)
            with p.open('rb') as f:
                _,n,props=header(f);rows=np.fromfile(f,dtype=np.dtype([(k,'<'+t) for k,t in props]),count=n)
            np.testing.assert_array_equal(rows['f_rest_0'],sh[:,1,0]);np.testing.assert_array_equal(rows['f_rest_8'],sh[:,1,1]);np.testing.assert_array_equal(rows['f_rest_23'],sh[:,8,2])
            b,s=read_sh_ply(p);np.testing.assert_array_equal(b,base);np.testing.assert_array_equal(s,sh)
            with self.assertRaisesRegex(ValueError,'Nonzero higher-order'):read_gaussians(p)
    def test_reject_invalid_degree_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.ply';base=np.zeros((1,14),'f')
            with self.assertRaises(ValueError):write_sh_ply(p,base,np.zeros((1,5,3),'f'))
            sh=np.zeros((1,9,3),'f');sh[0,1,0]=np.nan
            with self.assertRaises(ValueError):write_sh_ply(p,base,sh)

if __name__=='__main__':unittest.main()
