import {readFile,writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const scene=process.argv[2],viewsPath=process.argv[3]||'four-room/scenes/oblique-audit-v1/views.json';
assert(/^[\w-]+$/.test(scene));
const views=JSON.parse(await readFile(viewsPath,'utf8')).views,root=`four-room/scenes/${scene}/oblique-browser`;
await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:768,height:768,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/mesh-photo-room.html?scene='+scene});
 let ready=false;for(let i=0;i<120;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.meshReady')){ready=true;break}}assert(ready);
 for(const pose of views){
  const frame=await ev(`(()=>{const v=window.meshViewer,p=${JSON.stringify(pose)};v.camera.position.fromArray(p.center);v.camera.fov=90;v.camera.updateProjectionMatrix();v.controls.target.copy(v.camera.position);for(let k=0;k<3;k++)v.controls.target.setComponent(k,p.center[k]+p.basis[k][2]);v.controls.update();v.render();return {png:v.renderer.domElement.toDataURL(),position:v.camera.position.toArray(),inverse:v.camera.matrixWorldInverse.toArray()}})()`);
  assert(Math.hypot(...frame.position.map((x,k)=>x-pose.center[k]))<1e-6);
  // Three uses -Z forward and +Y up; CV evaluation uses +Z forward and +Y down.
  for(let r=0;r<3;r++)for(let c=0;c<4;c++)assert(Math.abs(frame.inverse[c*4+r]-(r? -1:1)*pose.viewmat[r][c])<1e-5,'Camera basis mismatch');
  await writeFile(`${root}/${pose.name}.png`,Buffer.from(frame.png.split(',')[1],'base64'));
 }
 assert.equal(errors.length,0,JSON.stringify(errors));await writeFile(root+'/check.json',JSON.stringify({passed:true,views:views.map(v=>v.name),errors},null,2));
 console.log('Oblique browser views passed',views.length);
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
