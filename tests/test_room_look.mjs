import assert from 'node:assert/strict';
import {PerspectiveCamera,Vector3} from '../three.js';
import {setRoomLook,installRoomLook} from '../room-look.js';
const camera=new PerspectiveCamera();camera.position.set(3,4,5);camera.up.set(.1,.2,1).normalize();
const controls={target:new Vector3(4,4,5),update(){camera.position.add(new Vector3(.2,-.4,.6))},dispatchEvent(){}};
const element={addEventListener(){},removeEventListener(){}};
const dispose=installRoomLook({camera,controls,element});
assert.equal(controls.minPolarAngle,0);assert.equal(controls.maxPolarAngle,Math.PI);assert.equal(controls.enableRotate,false);
for(const direction of [camera.up,camera.up.clone().negate(),new Vector3(1,0,0)]){
 const before=camera.position.clone();setRoomLook(camera,controls,direction);
 assert(camera.position.distanceTo(before)<1e-12,'Eye moved during head rotation');
 assert(Number.isFinite(camera.quaternion.w));
}
assert.throws(()=>setRoomLook(camera,controls,new Vector3()),/nonzero/);
dispose();console.log('Room look tests passed');
