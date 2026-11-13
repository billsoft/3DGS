"""Recoloring must preserve unusual source quaternions and ancillary PLY data."""
import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from native_photo_refit import write_native_colors

class NativePhotoIO(unittest.TestCase):
    def test_only_dc_bytes_change(self):
        for endian,fmt in [('<','binary_little_endian'),('>','binary_big_endian')]:
            with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as tmp:
                fields=['x','f_dc_0','rot_0','f_dc_1','opacity','f_dc_2','extra']
                a=np.zeros(2,dtype=[(k,endian+'f4') for k in fields])
                a['rot_0']=[.999991,2.];a['extra']=[-17,13];a['opacity']=[-5,7]
                head=('ply\nformat '+fmt+' 1.0\ncomment preserve me\nelement vertex 2\n'+''.join('property float '+k+'\n' for k in fields)+'end_header\n').encode()
                source=Path(tmp)/'in.ply';out=Path(tmp)/'out.ply'
                source.write_bytes(head+a.tobytes()+b'preserved tail')
                colors=np.array([[1,2,3],[4,5,6]],dtype='f4')
                write_native_colors(source,out,colors)
                data=out.read_bytes();self.assertEqual(data[:len(head)],head);self.assertTrue(data.endswith(b'preserved tail'))
                b=np.frombuffer(data,dtype=a.dtype,count=2,offset=len(head))
                for k in fields:
                    expected=colors[:,int(k[-1])] if k.startswith('f_dc_') else a[k]
                    np.testing.assert_array_equal(b[k],expected)
                self.assertEqual(source.read_bytes(),head+a.tobytes()+b'preserved tail')
    def test_reject_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'in.ply'
            src.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nend_header\n'+bytes(12))
            with self.assertRaisesRegex(ValueError,'Invalid color'):
                write_native_colors(src,Path(tmp)/'out.ply',np.array([[np.nan,0,0]]))

if __name__=='__main__':unittest.main()
