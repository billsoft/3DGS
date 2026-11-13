import sys,unittest,itertools,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from texture_graphcut import optimize,energy,expand

@unittest.skipUnless(importlib.util.find_spec('maxflow'),'Optional PyMaxflow is not installed')
class GraphTests(unittest.TestCase):
    def test_binary_matches_exhaustive_minimum(self):
        rng=np.random.default_rng(30)
        a=np.array([0,1,2,3,0,1]);b=np.array([1,2,3,4,4,3])
        for _ in range(8):
            unary=rng.uniform(0,3,(5,2));w=rng.uniform(.1,2,len(a))
            out,h=optimize(unary,a,b,w)
            best=min(energy(np.array(x),unary,a,b,w) for x in itertools.product(range(2),repeat=5))
            self.assertAlmostEqual(energy(out,unary,a,b,w),best,places=7)
    def test_each_expansion_is_optimal_binary_move(self):
        rng=np.random.default_rng(37);a=np.array([0,1,2,3,0]);b=np.array([1,2,3,4,4]);w=np.ones(5)
        unary=rng.uniform(0,4,(5,3));labels=np.array([0,1,2,0,1])
        for alpha in range(3):
            out=expand(labels,unary,a,b,w,alpha)
            best=min(energy(np.where(x,alpha,labels),unary,a,b,w) for x in itertools.product([False,True],repeat=5))
            self.assertAlmostEqual(energy(out,unary,a,b,w),best,places=7)
    def test_unseen_photo_cannot_win(self):
        unary=np.array([[0,1e8],[0,1],[0,1]],float);a=np.array([0,1]);b=np.array([1,2]);w=np.ones(2)*20
        labels,h=optimize(unary,a,b,w);self.assertEqual(labels[0],0);self.assertTrue(np.all(np.diff(h)<0))
if __name__=='__main__':unittest.main()
