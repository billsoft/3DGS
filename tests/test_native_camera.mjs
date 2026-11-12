import assert from 'node:assert/strict';
import {Matrix4,Vector3,Quaternion,PerspectiveCamera,Euler} from '../three.js';
import {nativeTransform,syncNativeCamera} from '../native-camera.js';
import {anchorPolicy} from '../native-anchor-policy.js';
for(const scale of [.2,1,5]){
 const m=new Matrix4().compose(new Vector3(3,-4,2),new Quaternion().setFromEuler(new Euler(.3,-.8,.2)),new Vector3(scale,scale,scale));
 const rows=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m.elements[c*4+r]));
 const transform=nativeTransform(rows),world=new PerspectiveCamera(70,1.5,.1,100),local=new PerspectiveCamera();
 world.position.set(2,-1,4);world.lookAt(1,2,0);world.updateMatrixWorld(true);syncNativeCamera(world,local,transform);
 for(const p of [new Vector3(0,0,0),new Vector3(1,2,-1),new Vector3(-3,2,1)]){
  const a=p.clone().applyMatrix4(m).project(world),b=p.clone().project(local);
  assert(a.distanceTo(b)<1e-9,'Similarity conversion changed screen projection');
 }
}
for(const count of [2,3,4,6,9]){
 const centers=Array.from({length:count},(_,i)=>[Math.cos(i/count*2*Math.PI)*4,Math.sin(i/count*2*Math.PI)*4,0]);
 centers.forEach((c,i)=>{const p=anchorPolicy(c,centers);assert.equal(p.source,i);assert.equal(p.weight,1)});
 const mid=centers[0].map((x,k)=>(x+centers[1][k])/2);assert.equal(anchorPolicy(mid,centers).weight,0);
}
console.log('Native camera projection invariance and 2/3/4/6/9-station policies passed');
