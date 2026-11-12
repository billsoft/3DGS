import unittest,tempfile,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from core import initialize,module
choose_labels=module('variable_scene_ownership',Path(__file__).resolve().parents[1]/'scene_fusion/ownership.py').choose_labels
radial_map=module('variable_scene_warp',Path(__file__).resolve().parents[1]/'scene_fusion/warp.py').radial_map

class SceneTests(unittest.TestCase):
    def test_radial_warp_preserves_camera_ray_and_correct_jacobian(self):
        rng=np.random.default_rng(42);control=rng.normal(size=(16,3));value=rng.normal(0,.05,16)
        center=np.array([.1,.2,.3]);points=rng.normal(size=(3,3))
        mapped,J,_=radial_map(points,center,control,value,.8)
        before=points-center;after=mapped-center
        np.testing.assert_allclose(before/np.linalg.norm(before,axis=1)[:,None],after/np.linalg.norm(after,axis=1)[:,None],atol=1e-10)
        for axis in range(3):
            eps=np.eye(3)[axis]*1e-5
            plus=radial_map(points+eps,center,control,value,.8)[0]
            minus=radial_map(points-eps,center,control,value,.8)[0]
            np.testing.assert_allclose((plus-minus)/2e-5,J[:,:,axis],atol=1e-7)

    def test_variable_pair_counts_and_cache_identity(self):
        for count in [2,4,6,9]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);inputs=root/'in';inputs.mkdir();out=root/'out'
                for i in range(count):
                    (inputs/f'{i}.png').write_bytes(b'photo');(inputs/f'{i}.ply').write_bytes(b'ply')
                meta=initialize(inputs,out);self.assertEqual(meta['count'],count)
                self.assertEqual(initialize(inputs,out),meta)
                (inputs/'0.ply').write_bytes(b'changed')
                with self.assertRaises(ValueError):initialize(inputs,out)

    def test_ownership_never_assigns_unsupported_source(self):
        for count in [2,4,6,9]:
            scores=np.zeros((count*2,count),dtype='f')
            for i in range(len(scores)):scores[i,i%count]=1
            neighbors=np.tile(np.arange(len(scores)),(len(scores),1))
            labels,_=choose_labels(scores,neighbors,penalty=100)
            np.testing.assert_array_equal(labels,np.arange(len(scores))%count)

    def test_quality_can_override_nearest_camera(self):
        scores=np.array([[.1,1],[.2,1],[.1,1]],dtype='f');neighbors=np.tile(np.arange(3),(3,1))
        labels,_=choose_labels(scores,neighbors)
        np.testing.assert_array_equal(labels,[1,1,1])

if __name__=='__main__':unittest.main()
