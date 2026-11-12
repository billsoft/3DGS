"""Check native-Windows gsplat 2D surface rasterization and gradients."""
import torch
from gsplat import rasterization_2dgs

torch.set_num_threads(4)
means=torch.tensor([[0.,0.,2.]],device='cuda',requires_grad=True)
quats=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True)
scales=torch.tensor([[.4,.4,.01]],device='cuda',requires_grad=True)
opacity=torch.tensor([.99],device='cuda',requires_grad=True)
colors=torch.tensor([[.6,.6,.6]],device='cuda',requires_grad=True)
K=torch.tensor([[[32.,0.,32.],[0.,32.,32.],[0.,0.,1.]]],device='cuda')
result=rasterization_2dgs(means,quats,scales,opacity,colors[None],torch.eye(4,device='cuda')[None],K,64,64,render_mode='RGB+ED',distloss=True,depth_mode='median')
rgb,alpha,normal,surf,dist,median,meta=result
loss=rgb[...,:3].mean()+dist.mean()+.1*(normal-surf).square().mean()
loss.backward()
assert all(torch.isfinite(x.grad).all() for x in [means,quats,scales,opacity,colors])
assert abs(float(median[0,32,32,0])-2)<1e-4
print('2D surface forward/backward passed',[(tuple(x.shape) if isinstance(x,torch.Tensor) else type(x).__name__) for x in result],float(loss))
