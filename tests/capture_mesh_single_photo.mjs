// Diagnostic only: project ONE existing photograph over unchanged geometry.
// Deliberately ignores occlusion support; never publish this as a fused mesh.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const [scene,cameraJson,station='3',face='4']=process.argv.slice(2);
assert(/^[\w-]+$/.test(scene));
const cameras=JSON.parse(await readFile(cameraJson,'utf8'));
const i=Number(station),root=`four-room/scenes/${scene}/diagnostics`;
await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.includes('7870'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:`http://127.0.0.1:7870/mesh-photo-room.html?scene=${scene}`});
 let ready=false;for(let n=0;n<90;n++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.meshReady')){ready=true;break}}assert(ready);
 const data=await ev(`(()=>{
  const v=window.meshViewer;v.inspect(${i},${Number(face)});v.mode('photo');
  const C=${JSON.stringify(cameras.C[i])},Q=${JSON.stringify(cameras.Q[i])};
  const g=v.mesh.geometry,p=g.attributes.position,u=g.attributes.uv;
  for(let k=0;k<p.count;k++){
   const d=[p.getX(k)-C[0],p.getY(k)-C[1],p.getZ(k)-C[2]],q=[0,1,2].map(j=>d.reduce((s,x,l)=>s+x*Q[l][j],0));
   u.setXY(k,Math.atan2(-q[1],q[0])/(2*Math.PI)+.5,1-Math.atan2(Math.hypot(q[0],q[1]),q[2])/Math.PI);
  }
  for(let k=0;k<p.count;k+=3){const a=[u.getX(k),u.getX(k+1),u.getX(k+2)];if(Math.max(...a)-Math.min(...a)>.5)for(let j=0;j<3;j++)if(a[j]<.5)u.setX(k+j,a[j]+1)}
  u.needsUpdate=true;
  v.mesh.material=v.mesh.material[v.meta.groups.findIndex(g=>g.source===${i})];
  v.render();return v.renderer.domElement.toDataURL();
 })()`);
 await writeFile(`${root}/source-${i+1}-face-${face}-single-photo-diagnostic.png`,Buffer.from(data.split(',')[1],'base64'));
 console.log('Single-photo diagnostic only',scene,i,face);
}finally{ws.close()}
