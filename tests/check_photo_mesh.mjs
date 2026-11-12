import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const scene=process.argv[2]||'mesh-photo-six-v4',root='four-room/scenes/'+scene+'/browser';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.includes('7870'));
assert(tab);const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(m.params.args);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function shot(name){const data=await ev('window.meshViewer.render();window.meshViewer.renderer.domElement.toDataURL()');await writeFile(root+'/'+name+'.png',Buffer.from(data.split(',')[1],'base64'))}
try{
 await call('Runtime.enable');await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true});await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/mesh-photo-room.html?scene='+scene});await call('Page.bringToFront');
 let ready=false;for(let i=0;i<120;i++){await pause(1000);if(await ev('!!window.meshReady')){ready=true;break}}assert(ready);
 const meta=await ev('window.meshViewer.meta');const records=[];
 for(let i=0;i<meta.centers.length;i++)for(const face of [3,4,5]){
  const position=await ev(`window.meshViewer.inspect(${i},${face})`);assert(Math.hypot(...position.map((x,k)=>x-meta.centers[i][k]))<1e-6);
  await shot('source-'+(i+1)+'-face-'+face);records.push({source:i,face});console.log('Captured',i+1,face);
 }
 await ev('window.meshViewer.inspect(0,3)');
 for(const [a,b] of meta.midpoint_edges){const p=meta.centers[a].map((v,k)=>(v+meta.centers[b][k])/2);await ev(`(()=>{const v=window.meshViewer,old=v.camera.position.clone();v.camera.position.fromArray(${JSON.stringify(p)});v.controls.target.add(v.camera.position.clone().sub(old));v.controls.update();v.render()})()`);await shot(`midpoint-${a+1}-${b+1}`)}
 await ev('window.meshViewer.inspect(0,3)');await ev('window.meshViewer.mode("solid")');await shot('geometry');await ev('window.meshViewer.mode("owner")');await shot('ownership');await ev('window.meshViewer.mode("photo")');
 assert.equal(errors.length,0,JSON.stringify(errors));
 await writeFile(root+'/check.json',JSON.stringify({passed:true,triangles:meta.triangles,sources:meta.stations.length,views:records,midpoints:meta.midpoint_edges.length,errors},null,2));console.log('Mesh browser checks passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
