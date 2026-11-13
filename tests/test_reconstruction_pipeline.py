import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from camera_graph import build_graph
from reconstruct import make_plan
from teacher_cache import TeacherCache
from run_state import RunState
from core import look

def registration(ids,edges):
    return dict(stations=ids,edges={f'{ids[a]}-{ids[b]}':dict(accepted=True,global_held_inliers=10) for a,b in edges})

class ReconstructionTests(unittest.TestCase):
    def test_sparse_branch_graph_any_count_and_permutation(self):
        rng=np.random.default_rng(913)
        for n in (2,3,5,8,11):
            C=rng.normal(size=(n,3));ids=[f'cam-{i:04d}' for i in range(n)]
            edges=[(0,i) for i in range(1,n)] # Star, not capture ring.
            graph=build_graph(C,ids,registration(ids,edges))
            self.assertEqual(set(map(tuple,graph['edges'])),set(edges))
            order=rng.permutation(n);inverse=np.argsort(order);new_ids=[ids[i] for i in order]
            remapped=[tuple(sorted((int(inverse[a]),int(inverse[b])))) for a,b in edges]
            result=build_graph(C[order],new_ids,registration(new_ids,remapped))
            actual={tuple(sorted((int(order[a]),int(order[b])))) for a,b in result['edges']}
            self.assertEqual(actual,set(edges))

    def test_close_but_unmatched_cameras_are_not_connected(self):
        ids=['a','b','c','d'];C=np.array([[0,0,0],[10,0,0],[.001,0,0],[10.001,0,0]])
        with self.assertRaisesRegex(ValueError,'Disconnected'):
            build_graph(C,ids,registration(ids,[(0,1),(2,3)]))
        r=registration(ids,[(0,1),(1,2),(2,3)]);r['graph']={'edge_weights':{'1-2':.01}}
        with self.assertRaisesRegex(ValueError,'Disconnected'):build_graph(C,ids,r)

    def test_plan_preserves_sequence_and_avoids_scene_constants(self):
        for n in (2,3,8):
            plan=make_plan(Path('new-job'),[f'cam-{i:04d}' for i in range(n)])
            names=[s['name'] for s in plan]
            self.assertLess(names.index('export'),names.index('capture'))
            self.assertLess(names.index('depth-smooth'),names.index('guide'))
            for mode in ('adaptive','smooth'):
                self.assertLess(names.index(mode+'-lowpass'),names.index(mode+'-texture'))
            self.assertEqual(len(next(s for s in plan if s['name']=='prepare')['outputs']),12*n)
            payload=json.dumps(plan)
            for forbidden in ('six-v','geometry-quads','native-source-six','cam-0006.ply'):self.assertNotIn(forbidden,payload)
        self.assertNotIn('smooth-texture',[s['name'] for s in make_plan('job',['a','b'],mode='adaptive')])

    def test_teacher_lru_is_bounded_for_many_sources(self):
        loaded=[]
        def load(p):loaded.append(p.name);return object()
        cache=TeacherCache(Path('data'),20,loader=load)
        first=cache[0];self.assertIs(cache[0],first)
        for i in range(1,20):cache[i];self.assertLessEqual(len(cache.models),2)
        cache[0];self.assertEqual(loaded.count('teacher-0.npy'),2)
        with self.assertRaises(IndexError):cache[20]

    def test_conflict_partition_is_default_with_explicit_legacy_comparison(self):
        for mode in ('adaptive','smooth','both'):
            default=make_plan('new-job',['a','b','c'],mode=mode)
            legacy=make_plan('new-job',['a','b','c'],mode=mode,conflict_regions=False)
            for current,old in zip(default,legacy):
                if current['name'].endswith('-texture'):
                    self.assertIn('--conflict-regions',current['args'])
                    self.assertNotIn('--conflict-regions',old['args'])
                    self.assertEqual([a for a in current['args'] if a!='--conflict-regions'],old['args'])
                else:self.assertEqual(current,old)

    def test_directory_checkpoint_detects_nested_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'stage';calls=[]
            def work():folder.mkdir(exist_ok=True);(folder/'image').write_text('correct');calls.append(1)
            s=RunState(root,'code');s.stage('stage',[],[folder],work)
            s=RunState(root,'code',True);s.stage('stage',[],[folder],work);self.assertEqual(len(calls),1)
            (folder/'later-diagnostic').write_text('not a stage output')
            s=RunState(root,'code',True);s.stage('stage',[],[folder],work);self.assertEqual(len(calls),1)
            (folder/'image').write_text('broken')
            s=RunState(root,'code',True);s.stage('stage',[],[folder],work);self.assertEqual(len(calls),2)

    def test_vertical_midpoint_look_is_finite(self):
        b=look(np.zeros(3),np.array([0.,0,-1]),np.array([0.,0,1]))
        self.assertTrue(np.isfinite(b).all());np.testing.assert_allclose(b.T@b,np.eye(3))

if __name__=='__main__':unittest.main()
