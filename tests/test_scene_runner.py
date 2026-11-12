import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from input_io import FIELDS,read_gaussians,normalize
from core import discover
from run_state import RunState
from quality import assess

class RunnerTests(unittest.TestCase):
    def test_ply_formats_reordered_properties_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.ply';names=['nx',*reversed(FIELDS)]
            expected=np.zeros((3,14),dtype='f');expected[:,:3]=np.arange(9).reshape(3,3);expected[:,10]=1
            values=np.column_stack([np.zeros(3),expected[:,::-1]])
            for fmt,endian in [('ascii','<'),('binary_little_endian','<'),('binary_big_endian','>')]:
                header='ply\nformat '+fmt+' 1.0\nelement vertex 3\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n'
                with path.open('wb') as f:
                    f.write(header.encode())
                    if fmt=='ascii':np.savetxt(f,values)
                    else:f.write(values.astype(endian+'f4').tobytes())
                np.testing.assert_array_equal(read_gaussians(path),expected)
            path.write_bytes(path.read_bytes()[:-4])
            with self.assertRaisesRegex(ValueError,'Truncated'):read_gaussians(path)

    def test_units_and_large_offset_normalize_consistently(self):
        rng=np.random.default_rng(4);a=np.zeros((1000,14),dtype='f');a[:,:3]=rng.normal(size=(1000,3));a[:,7:10]=-4;a[:,10]=1
        reference,_=normalize(a)
        for scale in [.01,100]:
            b=a.copy();b[:,:3]*=scale;b[:,7:10]+=np.log(scale)
            result,_=normalize(b);np.testing.assert_allclose(result,reference,atol=2e-6)
        b=a.copy();b[:,:3]+=10000
        result,meta=normalize(b);self.assertGreater(np.linalg.norm(meta['origin']),1000)
        self.assertLess(np.linalg.norm(np.median(result[:,:3],axis=0)),.01)

    def test_rejects_mesh_nonfinite_and_nonzero_higher_sh(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.ply'
            for names,values,error in [(['x','y','z'],[0,0,0],'missing'),
                    (FIELDS,[float('nan'),*([0]*9),1,0,0,0],'Non-finite'),
                    ([*FIELDS,'f_rest_0'],[*([0]*10),1,0,0,0,.1],'higher-order')]:
                header='ply\nformat ascii 1.0\nelement vertex 1\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n'
                path.write_text(header+' '.join(map(str,values)),encoding='ascii')
                with self.assertRaisesRegex(ValueError,error):read_gaussians(path)

    def test_unicode_reserved_names_and_subsets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['final','机位 甲','10','2']:
                (root/(name+'.PLY')).write_bytes(b'x');(root/(name+'.PNG')).write_bytes(b'y')
            items=discover(root)['stations'];self.assertEqual(len(items),4)
            self.assertEqual([x['id'] for x in items],['cam-0001','cam-0002','cam-0003','cam-0004'])
            self.assertEqual(discover(root,['final','机位 甲'])['count'],2)
            with self.assertRaises(ValueError):discover(root,['missing','2'])

    def test_checkpoint_content_configuration_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=root/'data';calls=[]
            def work():calls.append(1);artifact.write_text('correct')
            state=RunState(root,'code');state.stage('a',['arg'],[artifact],work);state.complete()
            state=RunState(root,'code',True);state.stage('a',['arg'],[artifact],work);self.assertEqual(len(calls),1)
            artifact.write_text('corrupt')
            state=RunState(root,'code',True);state.stage('a',['arg'],[artifact],work);self.assertEqual(len(calls),2)
            state=RunState(root,'code',True);state.stage('a',['changed'],[artifact],work);self.assertEqual(len(calls),3)
            def failure():raise RuntimeError('diagnostic')
            with self.assertRaises(RuntimeError):state.stage('b',[],[],failure)
            data=json.loads((root/'run-state.json').read_text());self.assertEqual(data['status'],'failed');self.assertIn('diagnostic',data['error'])

    def test_quality_rejects_local_damage_despite_good_average(self):
        bundle=dict(held_tracks=20,before_held=dict(median_equivalent_px=1),after_held=dict(median_equivalent_px=.5))
        native=dict(summary=dict(native_ssim=.9,fused_ssim=.91,newly_low_alpha_pixels=0,native_high_alpha_pixels=10000),views=[dict(native=dict(ssim=.9,mask_pixels=10000),fused=dict(ssim=.91,holes=0))])
        evaluation=dict(summary=dict(candidate_mean_ssim=.91,baseline_mean_ssim=.85))
        self.assertTrue(assess(bundle,native,evaluation)['passed'])
        bad=copy.deepcopy(native);bad['views'][0]['fused']['ssim']=.7
        self.assertFalse(assess(bundle,bad,evaluation)['passed'])
        bad=copy.deepcopy(native);bad['views'][0]['fused']['holes']=101
        self.assertFalse(assess(bundle,bad,evaluation)['passed'])
        bad=copy.deepcopy(bundle);bad['after_held']['median_equivalent_px']=4
        self.assertFalse(assess(bad,native,evaluation)['passed'])
        bad=copy.deepcopy(native);bad['views'][0]['fused']['ssim']=float('nan')
        self.assertFalse(assess(bundle,bad,evaluation)['passed'])

if __name__=='__main__':unittest.main()
