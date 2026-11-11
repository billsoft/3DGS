"""Image-driven sampling and tangent Gaussian footprints, without semantic regions."""
import cv2 as cv
import numpy as np

def sample_map(image, uv):
    out=[]
    for start in range(0,len(uv),20000):
        t=uv[start:start+20000]
        x=((t[:,0]%1)*image.shape[1]-.5).astype('f')[:,None]
        y=np.clip(t[:,1]*image.shape[0]-.5,0,image.shape[0]-1).astype('f')[:,None]
        out.append(cv.remap(image,x,y,cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,0])
    return np.concatenate(out) if out else np.empty((0,)+image.shape[2:])

def gradient_map(image):
    h,w=image.shape[:2];scale=min(1,3840/w)
    gray=cv.cvtColor(cv.resize(image,(round(w*scale),round(h*scale))),cv.COLOR_BGR2GRAY).astype('f')/255
    gray=cv.GaussianBlur(gray,(0,0),.65)
    return np.stack([cv.Sobel(gray,cv.CV_32F,1,0)*scale/8,
                     cv.Sobel(gray,cv.CV_32F,0,1)*scale/8],-1)

def choose_density(texture,base,gradient,image_shape,budget_factor=1.8):
    # Seven tests per triangle capture edges without assuming object identities.
    bary=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3]*3])
    probes=np.einsum('bk,nkj->nbj',bary,texture)
    g=sample_map(gradient,probes.reshape(-1,2)).reshape(len(texture),7,2)
    strength=np.clip((np.linalg.norm(g,axis=2).max(1)-.005)/.035,0,1)
    pixels=texture*np.array(image_shape[1::-1])
    span=np.linalg.norm(pixels-np.roll(pixels,1,axis=1),axis=2).max(1)
    target=np.minimum(base*2,np.maximum(base,np.ceil(span/2.5)))
    factor=1.
    while True:
        n=np.maximum(base,np.ceil(base+(target-base)*strength*factor)).astype(int)
        if np.sum(n.astype('d')**2)<=budget_factor*np.sum(base.astype('d')**2) or factor<.005:break
        factor*=.75
    limit=budget_factor*np.sum(base.astype('d')**2)
    if np.sum(n.astype('d')**2)>limit:
        extra=n.astype('d')**2-base.astype('d')**2
        order=np.argsort(-strength);keep=np.zeros(len(n),bool)
        keep[order]=np.cumsum(extra[order])<=limit-np.sum(base.astype('d')**2)
        n=np.where(keep,n,base)
    return n,strength

def texture_jacobian(xyz,texture,image_shape):
    e=xyz[:,1:]-xyz[:,:1]
    t=(texture[:,1:]-texture[:,:1])*np.array(image_shape[1::-1])
    # d(pixel coordinates)/d(world coordinates), restricted to triangle tangent.
    gram=e@e.transpose(0,2,1)
    return t.transpose(0,2,1)@np.linalg.pinv(gram,rcond=1e-10)@e

def edge_covariance(cov,jacobian,gradient):
    normal=np.einsum('ni,nij->nj',gradient,jacobian)
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
    strength=np.clip((np.linalg.norm(gradient,axis=1)-.005)/.035,0,1)
    beta=.30*strength
    # Contract across the image edge; preserve coverage along it and thickness.
    A=np.eye(3)[None]-beta[:,None,None]*normal[:,:,None]*normal[:,None,:]
    return A@cov@A.transpose(0,2,1),strength
