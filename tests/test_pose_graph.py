import sys, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from pose_graph import fit_graph, apply, bundle_gate, estimate_scale,validated_update


class PoseGraphTests(unittest.TestCase):
    def scene(self, n, units=1):
        rng=np.random.default_rng(129);poses=[np.eye(4)]
        for i in range(1,n):
            m=np.eye(4);m[:3,:3]=Rotation.from_rotvec(rng.normal(0,.2,3)).as_matrix()*rng.uniform(.7,1.5)
            m[:3,3]=rng.normal(size=3)*units;poses.append(m)
        edges={}
        for i in range(n):
            for j in range(i+1,n):
                world=rng.normal(size=(45,3))*units
                x=apply(np.linalg.inv(poses[j]),world);y=apply(np.linalg.inv(poses[i]),world)
                edges[i,j]=dict(m=np.linalg.inv(poses[i])@poses[j],x=x,y=y,
                                train=np.ones(45,bool),tol=.03*units,cells=15,
                                sample_cells=np.column_stack([np.arange(45)%15,np.zeros(45)]))
        return poses,edges

    def test_arbitrary_counts_units_and_scale(self):
        for n in [2,4,6,9]:
            for units in [.001,1000]:
                truth,edges=self.scene(n,units);result,_=fit_graph(edges,n)
                for a,b in zip(truth,result):
                    np.testing.assert_allclose(a[:3,:3],b[:3,:3],atol=1e-7)
                    np.testing.assert_allclose(a[:3,3]/units,b[:3,3]/units,atol=1e-7)

    def test_dense_bad_loop_does_not_dominate(self):
        truth,edges=self.scene(6)
        e=edges[0,5];e['y']=e['y']+np.array([2.,0,0]);e['m']=e['m'].copy();e['m'][:3,3]+=[2,0,0]
        e['cells']=4
        for name in ['x','y','train','sample_cells']:e[name]=np.repeat(e[name],50,axis=0)
        result,report=fit_graph(edges,6)
        self.assertLess(np.linalg.norm(result[5][:3,3]-truth[5][:3,3]),.02)
        self.assertLess(report['edge_weights']['0-5'],.01)

    def test_disconnected_graph_rejected(self):
        _,edges=self.scene(3)
        with self.assertRaisesRegex(ValueError,'Disconnected'):fit_graph({(0,1):edges[0,1]},3)

    def test_sparse_nonring_with_permuted_capture_order(self):
        truth,all_edges=self.scene(5)
        pairs=[(0,2),(1,2),(2,3),(3,4)]
        edges={e:all_edges[e] for e in pairs}
        result,_=fit_graph(edges,5)
        for a,b in zip(truth,result):np.testing.assert_allclose(a,b,atol=1e-7)
        order=[4,1,3,0,2];inv=np.argsort(order);changed={}
        for (i,j),e in edges.items():
            a,b=int(inv[i]),int(inv[j]);row=dict(e)
            if a>b:
                a,b=b,a;row.update(m=np.linalg.inv(e['m']),x=e['y'],y=e['x'])
            changed[a,b]=row
        result,_=fit_graph(changed,5)
        for k,m in enumerate(result):np.testing.assert_allclose(m,np.linalg.inv(truth[order[0]])@truth[order[k]],atol=1e-7)

    def test_heldout_gate_rejects_local_regression_and_failed_optimizer(self):
        before=[np.ones(20),np.ones(20)]
        self.assertTrue(bundle_gate(before,[np.ones(20)*.6]*2,True)['passed'])
        self.assertFalse(bundle_gate(before,[np.zeros(20),np.ones(20)*2],True)['passed'])
        self.assertFalse(bundle_gate(before,before,False)['passed'])
        self.assertFalse(bundle_gate(before,[np.ones(20),np.full(20,np.inf)],True)['passed'])

    def test_scale_balances_regions_and_rejects_ambiguous_evidence(self):
        # Thousands of biased features on one surface must not dominate 20 regions.
        ratios=np.r_[np.full(60,1.4),np.full(3000,2.8)]
        cells=np.r_[np.repeat(np.arange(20),3),np.full(3000,21)][:,None]
        scale,report=estimate_scale(ratios,cells)
        self.assertLess(abs(scale/1.4-1),.001)
        for units in [.001,1000]:
            value,_=estimate_scale(ratios*units,cells)
            self.assertAlmostEqual(value/units,scale,places=8)
        with self.assertRaisesRegex(ValueError,'concentrated'):estimate_scale(np.ones(30),np.zeros((30,2)))
        with self.assertRaisesRegex(ValueError,'inconsistent'):estimate_scale(np.r_[np.ones(20),np.ones(20)*4],np.arange(40)[:,None])

    def test_automatic_backtracking_preserves_gauge(self):
        C0=np.array([[0.,0,0],[2,0,0],[0,1,0]]);C=C0.copy();C[2,2]=1
        Q=np.tile(np.eye(3),(3,1,1))
        def check(c,q):return [np.ones(20),np.ones(20),np.ones(20)*(1+c[2,2])]
        cc,qq,report=validated_update(C0,Q,C,Q,check,True)
        self.assertEqual(report['fraction'],.25)
        self.assertAlmostEqual(np.linalg.norm(cc[1]-cc[0]),2)
        np.testing.assert_array_equal(cc[0],C0[0])

if __name__=='__main__':unittest.main()
