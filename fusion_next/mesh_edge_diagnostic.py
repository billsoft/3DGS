"""Ray-trace the exported textured mesh to isolate mesh versus splat artifacts."""
from common import *
from collider_audit import scene_for
from PIL import Image,ImageDraw

def run():
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    rows=np.fromfile(baseline/'mesh.bin','<f4').reshape(-1,3,5);xyz=rows[:,:,:3]
    meta=json.loads((baseline/'mesh.json').read_text());group=np.repeat(np.arange(len(meta['groups'])),[g['count']//3 for g in meta['groups']])
    scene,mesh=scene_for(xyz.reshape(-1,3),np.arange(len(xyz)*3).reshape(-1,3))
    images=[cv.imread(str(baseline/g['texture'])) for g in meta['groups']]
    size=480;views=[];up=np.array(rm.meta['up'])
    for i,v in enumerate(json.loads((ROOT/'four-room/surface-fusion-v4/inspection-views.json').read_text())[:2]):
        z=np.array(v['target'])-C[i];z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);views.append((i,np.stack([x,np.cross(z,x),z],1)))
    views += [(i,Q[i]@rm.f.prep.FACES[face]) for i,face in [(0,2),(1,2),(0,5),(1,1)]]
    yy,xx=np.indices((size,size));rays=np.stack([(xx+.5-size/2)/(size/2),(yy+.5-size/2)/(size/2),np.ones_like(xx)],-1).reshape(-1,3)
    for row,(i,basis) in enumerate(views):
        ray=np.ascontiguousarray(rays@basis.T,'f');starts=np.ascontiguousarray(np.tile(C[i],(len(ray),1)),'f')
        face=scene.run(starts,ray,query='INTERSECT');depth=scene.run(starts,ray,query='DISTANCE');valid=face>=0
        ids=np.flatnonzero(valid);f=face[valid];hit=starts[valid]+ray[valid]*depth[valid,None]
        e=xyz[f,1:]-xyz[f,:1];v=hit-xyz[f,0];gram=e@e.transpose(0,2,1)
        bc=np.einsum('nij,nj->ni',np.linalg.pinv(gram,rcond=1e-9),np.einsum('nki,ni->nk',e,v))
        bary=np.c_[1-bc.sum(1),bc];tx=np.einsum('nk,nkj->nj',bary,rows[f,:,3:])
        rgb=np.zeros((size*size,3),'u1')
        for k,image in enumerate(images):
            take=group[f]==k;rgb[ids[take]]=sample(image,tx[take])[:,::-1]
        previous=Image.open(OUT/f'comparison-{row}.jpg');sheet=Image.new('RGB',(size*4,size+30),(20,24,30));draw=ImageDraw.Draw(sheet)
        for col,label in enumerate(['photo','native','textured mesh','fused Gaussian']):draw.text((col*size+6,8),label,fill='white')
        sheet.paste(previous.crop((0,30,size*2,size+30)),(0,30));sheet.paste(Image.fromarray(rgb.reshape(size,size,3)),(size*2,30));sheet.paste(previous.crop((size*2,30,size*3,size+30)),(size*3,30))
        sheet.save(OUT/f'mesh-diagnostic-{row}.jpg',quality=97)
        print('mesh diagnostic',row,flush=True)

if __name__=='__main__':run()
