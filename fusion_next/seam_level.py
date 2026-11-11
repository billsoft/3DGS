"""Bounded low-frequency surface color offsets; never move geometry or blur texture."""
from common import *
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg

def run():
    geo=np.load(OUT/'geometry.npz');data=np.load(OUT/'ownership.npz')
    p,tri=geo['p'],geo['tri'];owner=data['owner'];colors=data['colors'];N=len(tri)
    a,b=adjacency(tri);xyz=p[tri];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-9)
    valid=(owner[a]<4)&(owner[b]<4)&((normal[a]*normal[b]).sum(1)>.85)
    a,b=a[valid],b[valid];ca=colors[a,owner[a]];cb=colors[b,owner[b]]
    same=owner[a]==owner[b];weight=.7*np.exp(-np.linalg.norm(ca-cb,axis=1)**2/.06**2)*same
    target=np.zeros((len(a),3));seams=np.flatnonzero(~same)
    # Evaluate both photographs at one common 3D location, not two face centers.
    mid=(data['center'][a[seams]]+data['center'][b[seams]])*.5
    ims=photos((3840,1920));left=np.zeros((len(seams),3));right=left.copy()
    for i in range(4):
        for ids,out in [(owner[a[seams]],left),(owner[b[seams]],right)]:
            mask=ids==i;out[mask]=correct_rgb(sample(ims[i],uv(project(mid[mask],i)))[:,::-1]/255,i)
    diff=right-left
    stable=(np.linalg.norm(diff,axis=1)<.16)&(np.linalg.norm(left-ca[seams],axis=1)<.07)&(np.linalg.norm(right-cb[seams],axis=1)<.07)
    used=seams[stable];seam_a=a[used];seam_b=b[used];weight[used]=8;target[used]=diff[stable]
    keep=weight>1e-5;a,b,weight,target=a[keep],b[keep],weight[keep],target[keep]
    degree=np.bincount(np.r_[a,b],weights=np.r_[weight,weight],minlength=N)
    A=diags(degree+.025)-coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(N,N)).tocsr()
    rhs=np.zeros((N,3));np.add.at(rhs,a,weight[:,None]*target);np.add.at(rhs,b,-weight[:,None]*target)
    result=[];status=[]
    for c in range(3):
        x,code=cg(A,rhs[:,c],M=diags(1/(degree+.025)),rtol=1e-5,maxiter=500);result.append(x);status.append(int(code))
    offset=np.clip(np.stack(result,1),-.045,.045)
    before=np.linalg.norm(diff[stable],axis=1);after=np.linalg.norm(diff[stable]-(offset[seam_a]-offset[seam_b]),axis=1)
    accepted=bool(all(s==0 for s in status) and len(before)>0 and np.median(after)<np.median(before))
    if not accepted:offset[:]=0
    np.save(OUT/'seam-offset.npy',offset.astype('f'))
    write_json('seam-level-report.json',dict(accepted=accepted,constraints=int(len(used)),median_before=float(np.median(before)),median_after=float(np.median(after)),maximum_rgb_offset=float(abs(offset).max()),solver_status=status,note='Bounded low-frequency surface RGB offsets; seam metric is fitting residual, not held-out visual quality. Mesh texture preview does not apply this PLY color correction.'))
    print('surface seam leveling',accepted,len(used),np.median(before),np.median(after),flush=True)

if __name__=='__main__':run()
