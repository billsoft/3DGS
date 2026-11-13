"""Extract a mesh from one accepted Gaussian model via rendered depths and Open3D.

Depth is generated, not measured. Raw input PLY depths are never reused here.
Camera metadata remains calibrated to the original panorama photographs.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np


def save_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def camera_intrinsics(o3d,size):
    # gsplat evaluates pixel centers at x+.5; Open3D backprojects integer x.
    return o3d.camera.PinholeCameraIntrinsic(size,size,size/2,size/2,size/2-.5,size/2-.5)


def check_tsdf_backend(o3d):
    """Fail early for a broken backend, independently of private scene data."""
    size=64
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.zeros((size,size,3),'u1')),
        o3d.geometry.Image(np.full((size,size),2.,'f')),
        depth_scale=1.,depth_trunc=30.,convert_rgb_to_intensity=False)
    v=o3d.pipelines.integration.ScalableTSDFVolume(.04,.12,o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd,camera_intrinsics(o3d,size),np.eye(4))
    mesh=v.extract_triangle_mesh()
    if len(mesh.triangles)<100 or np.max(abs(np.asarray(mesh.vertices)[:,2]-2))>.04:
        raise RuntimeError('Open3D failed the synthetic depth-plane check. Use the tested Open3D 0.19.0 environment.')


def capture(a):
    import torch,cv2 as cv
    from core import load,prep,viewmat
    from refit_render import load_scene,render
    meta=load(a.source);cameras=np.load(a.source/'aligned-cameras.npz');C,Q=cameras['C'],cameras['Q']
    model=load_scene(a.model);a.out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.source/'scene.json',a.out/'scene.json');shutil.copyfile(a.source/'aligned-cameras.npz',a.out/'aligned-cameras.npz')
    (a.out/'data-512').mkdir(exist_ok=True);shutil.copyfile(a.source/'data-512/dataset.json',a.out/'data-512/dataset.json')
    dataset=json.loads((a.source/'data-512/dataset.json').read_text('utf8'))
    expected=np.array(json.loads((a.model.parent/'data/dataset.json').read_text('utf8'))['centers'])
    if expected.shape!=C.shape or not np.allclose(C,expected,atol=1e-6):raise ValueError('Model and panorama camera frames differ')
    positions=[(s['id'],C[i],Q[i],True) for i,s in enumerate(meta['stations'])]
    for i,j in dataset['midpoint_edges']:positions.append((f'mid-{i}-{j}',(C[i]+C[j])/2,Q[i],False))
    frames=[];points=model['means'].detach().cpu().numpy();lo,hi=np.quantile(points,[.005,.995],axis=0)
    torch.set_num_threads(4)
    with torch.no_grad():
        for label,center,rotation,original in positions:
            folder=a.out/('warped-depth' if original else 'virtual-depth')/label;folder.mkdir(parents=True,exist_ok=True)
            for k,b in enumerate(prep.FACES):
                basis=rotation@b;v=dict(viewmat=viewmat(center,basis).tolist());rgb,alpha,_=render(model,v,a.size,depth=True)
                rgb=rgb.cpu().numpy();alpha=alpha.cpu().numpy();depth=rgb[:,:,3]
                np.savez_compressed(folder/f'depth-{k}.npz',depth=depth,alpha=alpha)
                cv.imwrite(str(folder/f'color-{k}.jpg'),np.uint8(np.clip(rgb[:,:,:3][:,:,::-1],0,1)*255),[cv.IMWRITE_JPEG_QUALITY,95])
                frames.append(dict(depth=str((folder/f'depth-{k}.npz').relative_to(a.out)),color=str((folder/f'color-{k}.jpg').relative_to(a.out)),viewmat=v['viewmat'],original_camera=original))
            print('Captured single-model depth',label,flush=True)
    save_json(a.out/'capture.json',dict(size=a.size,frames=frames,bounds=[lo.tolist(),hi.tolist()],model=str(a.model),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),synthetic_depth=True))


def integrate(a):
    import open3d as o3d
    check_tsdf_backend(o3d)
    meta=json.loads((a.source/'capture.json').read_text('utf8'));size=meta['size'];lo,hi=np.array(meta['bounds'])
    voxel=float((hi-lo).max()/a.resolution);trunc=voxel*3
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=trunc,color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    intrinsic=camera_intrinsics(o3d,size)
    records=[]
    frames=[f for f in meta['frames'] if not a.original_only or f['original_camera']]
    if not frames:raise ValueError('No selected depth frames')
    for i,f in enumerate(frames):
        z=np.load(a.source/f['depth']);d=z['depth'].astype('f');alpha=z['alpha']
        # Keep sharp discontinuities; do not interpolate or blur foreground depth
        # into background. TSDF handles the geometric integration itself.
        good=np.isfinite(d)&(d>.1)&(d<30)&(alpha>=.95)
        d[~good]=0
        color=np.zeros((size,size,3),'u1')
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(d),depth_scale=1.,depth_trunc=30.,convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intrinsic,np.array(f['viewmat']))
        records.append(float(good.mean()))
        if i%12==0:print('Open3D TSDF',i,'/',len(frames),'voxel',voxel,flush=True)
    mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    raw_faces=len(mesh.triangles)
    if raw_faces>a.max_faces:mesh=mesh.simplify_quadric_decimation(a.max_faces)
    mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
    p=np.asarray(mesh.vertices).astype('f');tri=np.asarray(mesh.triangles).astype('i')
    if not len(tri) or not np.isfinite(p).all():raise ValueError('Invalid reconstructed surface')
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'surface.npz',p=p,tri=tri)
    o3d.io.write_triangle_mesh(str(a.out/'surface.ply'),mesh)
    save_json(a.out/'surface-report.json',dict(method='Open3D ScalableTSDFVolume',open3d_version=o3d.__version__,voxel=voxel,truncation=trunc,depth_views=len(frames),original_only=a.original_only,raw_faces=raw_faces,faces=len(tri),vertices=len(p),valid_depth_fraction=records,synthetic_depth=True,model_sha256=meta['model_sha256']))
    print('Mesh',len(p),'vertices',len(tri),'faces',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capture','integrate']);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,default=1024);p.add_argument('--resolution',type=int,default=640);p.add_argument('--max-faces',type=int,default=1600000)
    p.add_argument('--original-only',action='store_true',help='Exclude synthetic midpoint cameras from geometric integration')
    a=p.parse_args();globals()[a.stage](a)
