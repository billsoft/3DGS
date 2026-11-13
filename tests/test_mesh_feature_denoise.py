import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from mesh_feature_denoise import denoise
from texture_patch_consensus import coherent_patches,reject_photo_outliers

def grid(n=18):
    x,y=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n));p=np.c_[x.ravel(),y.ravel(),np.zeros(n*n)]
    a=np.arange(n*n).reshape(n,n)[:-1,:-1].ravel();t=np.vstack([np.c_[a,a+1,a+n],np.c_[a+1,a+n+1,a+n]])
    return p,t

class DenoiseTests(unittest.TestCase):
    def test_photo_outlier_requires_majority_and_three_visible_sources(self):
        scores=np.ones((3,6));colors=np.ones((3,6,3))*.6;coverage=np.ones_like(scores)
        colors[0,0]=0 # five agreeing observations reject a black wrong projection
        colors[1,:3]=0 # equally plausible clusters must not be forced
        colors[2,0]=0;coverage[2,2:]=0 # two visible photos are insufficient
        out,report=reject_photo_outliers(scores,colors,coverage)
        self.assertEqual(out[0,0],0);np.testing.assert_array_equal(out[1:],scores[1:])
        self.assertEqual(report['affected_faces'],1)
    def test_noise_plane_improves_and_boundary_stays(self):
        p,t=grid();inside=(p[:,0]>0)&(p[:,0]<1)&(p[:,1]>0)&(p[:,1]<1)
        p[inside,2]=np.random.default_rng(17).normal(0,.003,inside.sum())
        q,r=denoise(p,t)
        self.assertLess(np.std(q[inside,2]),np.std(p[inside,2])*.8)
        np.testing.assert_allclose(q[~inside],p[~inside],atol=1e-7)
        self.assertLessEqual(r['max_displacement'],r['displacement_cap']+1e-8)
    def test_scale_equivariance(self):
        p,t=grid(10);p[:,2]=.002*np.sin(p[:,0]*60)
        a,_=denoise(p,t);b,_=denoise(p*7,t)
        np.testing.assert_allclose(a*7,b,atol=1e-6)
    def test_nearby_disconnected_sheets_do_not_mix(self):
        p,t=grid(8);q=p.copy();q[:,2]=.001
        pts=np.vstack([p,q]);tri=np.vstack([t,t+len(p)])
        result,_=denoise(pts,tri);np.testing.assert_allclose(result,pts,atol=1e-7)
    def test_patch_reassignment_respects_visibility(self):
        tri=np.zeros((5,3),int);owner=np.array([1,0,0,0,0]);neighbors=np.array([[1,2,3],[0,2,4],[0,1,4],[0,2,4],[1,2,3]])
        scores=np.ones((5,2));colors=np.zeros((5,2,3));good=np.ones(5,bool);area=np.ones(5)
        out,report=coherent_patches(tri,owner,scores,colors,good,neighbors,area,1,max_faces=1)
        self.assertEqual(out[0],0)
        scores[0,0]=0
        out,_=coherent_patches(tri,owner,scores,colors,good,neighbors,area,1,max_faces=1)
        self.assertEqual(out[0],1)
    def test_patch_different_surface_color_is_preserved(self):
        tri=np.zeros((5,3),int);owner=np.array([1,0,0,0,0]);neighbors=np.array([[1,2,3],[0,2,4],[0,1,4],[0,2,4],[1,2,3]])
        colors=np.zeros((5,2,3));colors[0,1]=1
        out,_=coherent_patches(tri,owner,np.ones((5,2)),colors,np.ones(5,bool),neighbors,np.ones(5),1,max_faces=1)
        self.assertEqual(out[0],1)

if __name__=='__main__':unittest.main()
