import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const [scene,station='1',face='4']=process.argv.slice(2);
assert(/^[\w-]+$/.test(scene));
const root=`four-room/scenes/${scene}/diagnostics`;await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.includes('7870'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:`http://127.0.0.1:7870/mesh-photo-room.html?scene=${scene}`});
 let ready=false;for(let i=0;i<90;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.meshReady')){ready=true;break}}assert(ready);
 await ev(`window.meshViewer.inspect(${Number(station)},${Number(face)})`);
 const pose=await ev('(()=>{const v=window.meshViewer;return {center:v.camera.position.toArray(),matrixWorld:v.camera.matrixWorld.toArray(),fov:v.camera.fov,aspect:v.camera.aspect,width:v.renderer.domElement.width,height:v.renderer.domElement.height}})()');
 await writeFile(`${root}/source-${Number(station)+1}-face-${face}-pose.json`,JSON.stringify(pose,null,2));
 for(const mode of ['photo','unknown','solid','owner']){
  const data=await ev(`(()=>{const v=window.meshViewer;v.mode('${mode==='unknown'?'photo':mode}');if('${mode}'==='unknown'){v.meta.groups.forEach((g,i)=>{if(g.source<0){const m=v.mesh.material[i];m.map=null;m.color.set(0xff00ff);m.needsUpdate=true}})}v.render();return v.renderer.domElement.toDataURL()})()`);
  await writeFile(`${root}/source-${Number(station)+1}-face-${face}-${mode}.png`,Buffer.from(data.split(',')[1],'base64'));
 }
 console.log('Diagnostic captures',scene,station,face);
}finally{ws.close()}
