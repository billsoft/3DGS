import json,subprocess,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from tour_distill import tour_weights,sample_views

class TourDistillationTests(unittest.TestCase):
    def test_camera_centers_and_transition(self):
        c=np.array([[0,0,0],[10,0,0]],float)
        for n in [1,2,6,9]:
            centers=np.c_[np.arange(n)*3,np.zeros((n,2))]
            for i,p in enumerate(centers):self.assertEqual(tour_weights(p,centers)[i],1)
        np.testing.assert_allclose(tour_weights([4.4,0,0],c),[1,0],atol=1e-14)
        np.testing.assert_allclose(tour_weights([5,0,0],c),[.5,.5])
        np.testing.assert_allclose(tour_weights([5.6,0,0],c),[0,1],atol=1e-14)
        with self.assertRaises(ValueError):tour_weights([0,0,0],c,.5)

    def test_matches_actual_player_policy_off_route_and_at_ties(self):
        centers=np.array([[0,0,0],[3,0,0],[3,3,1],[0,3,1]],float)
        positions=np.vstack([np.random.default_rng(1).uniform(-1,4,(32,3)),centers,[[1.5,0,0],[1.5,1.5,.5]]])
        payload=json.dumps(dict(centers=centers.tolist(),positions=positions.tolist()))
        code="import {tourPolicy} from './native-tour-policy.js'; const d=JSON.parse(process.argv[1]); console.log(JSON.stringify(d.positions.map(p=>tourPolicy(p,d.centers,.44).weights)));"
        result=subprocess.check_output(['node','--input-type=module','-e',code,payload],cwd=Path(__file__).resolve().parents[1],text=True)
        np.testing.assert_allclose([tour_weights(p,centers) for p in positions],json.loads(result),atol=1e-14)

    def test_heldout_positions_not_used_for_training_and_all_directions(self):
        centers=[[0,0,0],[3,0,0]];basis=np.eye(3).tolist()
        m=dict(centers=centers,midpoint_edges=[[0,1]],views=[dict(source=i,name=f'source-{i}-{k}',basis=basis) for i in range(2) for k in range(6)])
        views=sample_views(m,.44)
        train={tuple(v['center']) for v in views if v['role']=='train'};test={tuple(v['center']) for v in views if v['role']=='test'}
        self.assertFalse(train&test)
        for p in train|test:self.assertEqual(sum(tuple(v['center'])==p for v in views),6)
        for v in views:
            self.assertAlmostEqual(sum(v['weights']),1)
            np.testing.assert_allclose(np.array(v['viewmat'])@np.r_[v['center'],1],[0,0,0,1],atol=1e-12)

if __name__=='__main__':unittest.main()
