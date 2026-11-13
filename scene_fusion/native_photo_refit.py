"""One panorama per native PLY: optimize rendered color, never geometry/opacity.

The differentiable renderer handles visibility and splat footprints. Training
views and rotated checks share the same photograph; neither is new-view truth.
"""
import argparse,json,hashlib,shutil,time,io,zipfile,html
from pathlib import Path
import numpy as np,cv2 as cv
from scipy.spatial.transform import Rotation
from core import prep,viewmat,write_json
from input_io import read_gaussians,header,normalize


def publish_report(out,native,reports):
    links=''
    for r,entry in zip(reports,native['stations']):
        label=html.escape(r['id']);status='附色通过' if r['passed'] else '保留原始模型'
        links+=f'<h2>{label} — {status}</h2><p>照片 MAE {r["before_mae"]:.4f} → {r["after_mae"]:.4f}</p>'
        links+=f'<a href="{html.escape(entry["file"],quote=True)}" download>下载此机位 PLY</a><br>'
        for v in r['views']:
            title=v['view'].replace('face-4','向上').replace('face-5','向下')
            links+=f'<a href="{label}/{v["view"]}.jpg">{title}</a> '
    page='<meta charset="utf-8"><title>一对一原片附色对照</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto}a{display:inline-block;margin:8px}</style><h1>一张全景图对应一份 PLY</h1><p>每张对照从左到右：原片 / 原始 PLY / 重新附色。检查来自同一张照片，不是未见视点验证。局部细纹理、反光和几何缺陷仍可能保留。</p><p><a href="native-photo-models.zip">下载全部机位 PLY</a><a href="/native-tour-room.html?scene='+html.escape(out.name,quote=True)+'">连续漫游预览</a></p>'+links
    (out/'index.html').write_text(page,encoding='utf8')
    with zipfile.ZipFile(out/'native-photo-models.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for entry in native['stations']:archive.write(out/entry['file'],Path(entry['file']).name)
        archive.write(out/'native.json','native.json')


def write_native_colors(source,destination,dc):
    """Change DC fields only, preserving source header and every other PLY byte."""
    data=bytearray(Path(source).read_bytes());stream=io.BytesIO(data)
    fmt,count,props=header(stream);offset=stream.tell()
    if fmt not in ('binary_little_endian','binary_big_endian'):
        raise ValueError('Byte-preserving recoloring requires binary Gaussian PLY')
    endian='<' if fmt=='binary_little_endian' else '>'
    dtype=np.dtype([(n,endian+t) for n,t in props])
    original=np.ndarray(count,dtype=dtype,buffer=bytes(data),offset=offset)
    output=np.ndarray(count,dtype=dtype,buffer=data,offset=offset)
    if dc.shape!=(count,3) or not np.isfinite(dc).all():raise ValueError('Invalid color array')
    fields=['f_dc_0','f_dc_1','f_dc_2']
    for k,name in enumerate(fields):output[name]=dc[:,k]
    for name,_ in props:
        if name not in fields and original[name].tobytes()!=output[name].tobytes():raise ValueError('Non-color field changed: '+name)
    Path(destination).write_bytes(data)


def repack(a):
    """Upgrade completed experiment exports to exact source-field preservation."""
    scene=json.loads((a.scene/'scene.json').read_text('utf8'))
    native=json.loads((a.out/'native.json').read_text('utf8'))
    for item,entry in zip(scene['stations'],native['stations']):
        if item['id']!=entry['id']:raise ValueError('Station order mismatch')
        candidate=a.out/item['id']/'candidate.ply'
        dc=read_gaussians(candidate)[:,3:6]
        write_native_colors(item['ply'],candidate,dc)
        chosen=a.out/entry['file']
        shutil.copyfile(candidate if entry['photo_refit_passed'] else item['ply'],chosen)
        entry['sha256']=hashlib.sha256(chosen.read_bytes()).hexdigest()
    native['non_color_fields_byte_identical']=True
    write_json(a.out/'native.json',native)
    publish_report(a.out,native,json.loads((a.out/'photo-refit.json').read_text('utf8')))


def run(a):
    if a.repack_only:return repack(a)
    import torch
    from refit_render import load_scene,render,SH0
    from refit_train import structure_loss
    from refit_evaluate import metrics
    from PIL import Image,ImageDraw
    if a.out.exists():raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(71)
    scene=json.loads((a.scene/'scene.json').read_text('utf8'));native=json.loads((a.native/'native.json').read_text('utf8'))
    if [s['id'] for s in scene['stations']]!=[s['id'] for s in native['stations']]:raise ValueError('Station order mismatch')
    reports=[]
    for item,entry in zip(scene['stations'],native['stations']):
        folder=a.out/item['id'];folder.mkdir();source=a.scene/item['id'];start=time.time()
        if hashlib.sha256(Path(item['photo']).read_bytes()).hexdigest()!=item['photo_sha256']:raise ValueError('Photo changed after calibration')
        if hashlib.sha256(Path(item['ply']).read_bytes()).hexdigest()!=item['ply_sha256']:raise ValueError('PLY changed after calibration')
        raw=read_gaussians(Path(item['ply']));normalized,_=normalize(raw)
        if not np.array_equal(normalized,np.load(source/'original.npy')):raise ValueError('Normalized source cache mismatch')
        del normalized
        photo=cv.imdecode(np.fromfile(item['photo'],np.uint8),1)[:,:,::-1].copy()
        camera=np.load(source/'cache.npz');C,R=camera['C'],camera['R']
        model=load_scene(source/'original.npy');initial=model['colors'].clone();colors=torch.nn.Parameter(initial.clone());model['colors']=colors
        # A fixed cubemap lets colors overfit one rasterization footprint. Train
        # overlapping orientations so a turn of the head remains constrained.
        bases=[Rotation.from_euler('z',angle,degrees=True).as_matrix()@b
               for angle in (0,22.5,45,67.5) for b in prep.FACES]
        opt=torch.optim.Adam([colors],lr=.008);views=[dict(viewmat=viewmat(C,R.T@b).tolist()) for b in bases]
        truth=[torch.tensor(prep.render.photo_view(photo,b,a.size).copy(),device='cuda',dtype=torch.float32)/255 for b in bases]
        reused=a.reuse_candidates/item['id']/'candidate.ply' if a.reuse_candidates else None
        if reused is not None and reused.exists():
            previous=read_gaussians(reused);raw_reference=read_gaussians(Path(item['ply']))
            if previous.shape!=raw_reference.shape or not np.array_equal(previous[:,:3],raw_reference[:,:3]) or not np.allclose(previous[:,6:],raw_reference[:,6:],atol=1e-6,rtol=0):raise ValueError('Candidate geometry mismatch')
            with torch.no_grad():colors.copy_(torch.as_tensor(previous[:,3:6]*SH0+.5,device='cuda').clamp(0,1))
        for step in range(0 if reused is not None and reused.exists() else a.steps):
            k=step%len(views);opt.zero_grad(set_to_none=True);rgb,alpha,_=render(model,views[k],a.size)
            loss=.8*(rgb-truth[k]).abs().mean()+.2*structure_loss(rgb,truth[k])+.0002*(colors-initial).square().mean()
            loss+=.08*((rgb[1:]-rgb[:-1]-truth[k][1:]+truth[k][:-1]).abs().mean()+(rgb[:,1:]-rgb[:,:-1]-truth[k][:,1:]+truth[k][:,:-1]).abs().mean())
            if not torch.isfinite(loss):raise ValueError('Nonfinite photo loss')
            loss.backward();opt.step()
            with torch.no_grad():colors.clamp_(0,1)
            if step==a.steps//2:
                for g in opt.param_groups:g['lr']*=.3
            if step%300==0:print(item['id'],'step',step,'loss',float(loss.detach()),flush=True)
        candidate=colors.detach().clone();records=[]
        checks=[('face-'+str(k),b,a.eval_size) for k,b in enumerate(prep.FACES)]
        turn=Rotation.from_euler('z',11.25,degrees=True).as_matrix()
        checks += [('rotated-'+str(k),turn@b,a.eval_size) for k,b in enumerate(prep.FACES)]
        with torch.no_grad():
            for name,b,size in checks:
                v=dict(viewmat=viewmat(C,R.T@b).tolist());target=prep.render.photo_view(photo,b,size)/255.
                model['colors']=initial;before,alpha,_=render(model,v,size);model['colors']=candidate;after,aa,_=render(model,v,size)
                old=before.clamp(0,1).cpu().numpy();new=after.clamp(0,1).cpu().numpy();mask=alpha.cpu().numpy()
                record=dict(view=name,before=metrics(old,target,mask,mask),after=metrics(new,target,aa.cpu().numpy(),mask));records.append(record)
                sheet=Image.new('RGB',(size*3,size+25));draw=ImageDraw.Draw(sheet)
                for col,(label,img) in enumerate([('source photo',target),('original PLY',old),('photo recolored PLY',new)]):
                    sheet.paste(Image.fromarray(np.uint8(img.clip(0,1)*255)),(col*size,25));draw.text((col*size+5,5),label,fill='white')
                sheet.save(folder/(name+'.jpg'),quality=95)
        passed=all(r['after']['mae']<=r['before']['mae']+.001 and r['after']['ssim']>=r['before']['ssim']-.002 for r in records)
        before_mae=float(np.mean([r['before']['mae'] for r in records]));after_mae=float(np.mean([r['after']['mae'] for r in records]));passed=passed and after_mae<before_mae
        output=raw.copy()
        if len(raw)!=len(initial):raise ValueError('Point identity/count mismatch')
        output[:,3:6]=(candidate.cpu().numpy()-.5)/SH0
        if not np.array_equal(output[:,:3],raw[:,:3]) or not np.array_equal(output[:,6:],raw[:,6:]):raise ValueError('Geometry changed')
        write_native_colors(item['ply'],folder/'candidate.ply',output[:,3:6])
        selected=a.out/entry['file'];shutil.copyfile(folder/'candidate.ply' if passed else Path(item['ply']),selected)
        if not np.array_equal(read_gaussians(selected)[:,6:],raw[:,6:]):raise ValueError('Export changed geometry')
        report=dict(id=item['id'],passed=bool(passed),selected='photo-recolored' if passed else 'original',before_mae=before_mae,after_mae=after_mae,
                    photo_resolution=list(photo.shape[:2]),steps=a.steps,training_size=a.size,evaluation_size=a.eval_size,training_yaw_degrees=[0,22.5,45,67.5],check_yaw_degrees=[0,11.25],
                    geometry_opacity_exact=True,source_photo_sha256=item['photo_sha256'],source_ply_sha256=item['ply_sha256'],
                    views=records,seconds=time.time()-start,limitation='Same-photo fitting checks only; nearby-view geometric correctness is not guaranteed.')
        entry['sha256']=hashlib.sha256(selected.read_bytes()).hexdigest();entry['photo_refit_passed']=bool(passed)
        reports.append(report);write_json(folder/'report.json',report);write_json(a.out/'photo-refit.json',reports)
        print(item['id'],'accepted',passed,'photo MAE',before_mae,'->',after_mae,flush=True)
        del model,opt,colors,initial,candidate,truth;torch.cuda.empty_cache()
    native['original_files_byte_identical']=False;native['non_color_fields_byte_identical']=True;native['photo_recolored']=True;native['reference_native_scene']=a.native.name
    native['note']='Each native PLY refitted only to its own panorama; geometry and opacity unchanged. Failed candidates retain the original PLY.'
    write_json(a.out/'native.json',native)
    publish_report(a.out,native,reports)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--size',type=int,default=1536);p.add_argument('--eval-size',type=int,default=1024);p.add_argument('--repack-only',action='store_true');p.add_argument('--reuse-candidates',type=Path);run(p.parse_args())
