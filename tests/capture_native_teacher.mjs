// Task-owned browser CDP: freeze native spatial policy and capture exact cameras.
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=process.argv[2]||'four-room/scenes/native-teacher-audit-v1';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:768,height:768,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-tour-room.html?scene=native-source-six-v1&range=0.49'});
 let ready=false;for(let i=0;i<180;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.tourState?.ready')){ready=true;break}}assert(ready);
 await ev('document.querySelector("#rest").checked=false;document.querySelector("#rest").dispatchEvent(new Event("change"))');
 const n=await ev('window.tourMeta.stations.length'),sources=[...new Set([0,Math.min(3,n-1)])],records=[];
 for(const source of sources)for(const [face,fov,near] of [[3,90,false],[4,90,false],[5,90,false],[4,45,false],[5,90,true]]){
  await ev(`window.inspectTour(${source},${face})`);
  const state=await ev(`(async()=>{const v=window.roomViewer;v.camera.fov=${fov};v.camera.updateProjectionMatrix();${near?`const c=window.tourMeta.centers,other=(${source}+1)%c.length,old=v.camera.position.clone();v.camera.position.fromArray(c[${source}].map((x,k)=>.9*x+.1*c[other][k]));v.controls.target.add(v.camera.position.clone().sub(old));v.controls.update();`:''}return await window.flushTour()})()`);
  assert.equal(state.weights[source],1);assert.equal(state.core,.49);assert.equal(state.usesFusedModel,false);
  const frame=await ev(`(()=>{const v=window.tourSlots.get(${source}).viewer,c=v.camera,w=window.roomViewer.camera;v.render();return {png:v.renderer.domElement.toDataURL(),localInverse:c.matrixWorldInverse.toArray(),worldInverse:w.matrixWorldInverse.toArray(),projection:c.projectionMatrix.toArray(),position:w.position.toArray(),width:v.renderer.domElement.width,height:v.renderer.domElement.height}})()`);
  const name=`source-${source+1}-${face}-fov${fov}-${near?'near':'anchor'}`;await writeFile(root+'/'+name+'.png',Buffer.from(frame.png.split(',')[1],'base64'));delete frame.png;
  records.push({name,source,face,fov,near,...frame});console.log('Captured',name);
 }
 assert.equal(errors.length,0,JSON.stringify(errors));await writeFile(root+'/captures.json',JSON.stringify({passed:true,core:.49,records,errors},null,2));
 await ev('window.inspectTour(0,3)');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
