import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_next'))
from edge_detail import edge_covariance,choose_density,gradient_map,texture_jacobian

class DetailTests(unittest.TestCase):
    def test_contracts_across_edge_only(self):
        cov=np.diag([4.,9.,.001])[None];j=np.array([[[1,0,0],[0,1,0]]])
        out,_=edge_covariance(cov,j,np.array([[.1,0]]))
        np.testing.assert_allclose(out[0].diagonal(),[4*.7**2,9,.001])
        self.assertTrue((np.linalg.eigvalsh(out)>0).all())

    def test_constant_color_preserves_covariance_and_density(self):
        im=np.full((64,128,3),100,'u1');grad=gradient_map(im)
        tex=np.array([[[.1,.1],[.7,.1],[.1,.7]]]);base=np.array([3])
        n,_=choose_density(tex,base,grad,im.shape[:2]);np.testing.assert_array_equal(n,base)
        c=np.eye(3)[None];out,_=edge_covariance(c,np.zeros((1,2,3)),np.zeros((1,2)))
        np.testing.assert_allclose(out,c)

    def test_jacobian_is_tangent(self):
        xyz=np.array([[[0.,0,2],[2,0,2],[0,3,2]]]);tex=np.array([[[0.,0],[1,0],[0,1]]])
        j=texture_jacobian(xyz,tex,(300,200))
        np.testing.assert_allclose(j,[[[100,0,0],[0,100,0]]])

    def test_discrete_density_respects_budget(self):
        gradient=np.ones((20,40,2),dtype='f')*.1
        texture=np.tile(np.array([[[.1,.1],[.9,.1],[.1,.9]]]),(10,1,1))
        base=np.ones(10,dtype=int)
        n,_=choose_density(texture,base,gradient,(20,40))
        self.assertLessEqual(float(np.sum(n**2)),1.8*np.sum(base**2))
        self.assertTrue((n>=base).all())

if __name__=='__main__':unittest.main()
