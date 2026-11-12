// Actual browser verification in the existing task-owned CDP tab.
import {writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=process.argv[2]||'four-room/scenes/native-source-six-v1/browser';await mkdir(root,{recursive:true});
const tabs=await(await fetch('http://localhost:19346/json/list')).json(),tab=tabs.find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
assert(tab);const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-scene-room.html?scene=native-source-six-v1'});
 let ready=false;for(let i=0;i<150;i++){await pause(1000);if(await ev('!!window.nativeState?.ready&&window.nativeState.baseReady')){ready=true;break}}
 assert(ready,'Native page failed to finish loading');const meta=await ev('window.nativeMeta');const records=[];
 console.log('Loaded',meta.stations.length,'sources');
 await ev('window.roomViewer.camera.fov=90;window.roomViewer.camera.updateProjectionMatrix()');
 for(let s=0;s<meta.stations.length;s++){
  for(let face=0;face<6;face++){
   const state=await ev(`window.inspectNative(${s},${face})`);
   assert.equal(state.source,s);assert.equal(state.weight,1);assert.deepEqual(state.visibleSources,[s]);
   assert(Math.hypot(...state.worldPosition.map((x,k)=>x-meta.centers[s][k]))<1e-5);
   assert(Math.hypot(...state.localPosition.map((x,k)=>x-meta.stations[s].local_capture[k]))<1e-5);
   const checked=await ev("(()=>{const v=window.nativeViewer;return {sorted:v.splatSortCount,visible:v.splatRenderCount,layerOpacity:getComputedStyle(document.querySelector('#native')).opacity,localPosition:v.camera.position.toArray()}})()");
   assert.equal(checked.layerOpacity,'1');assert.equal(checked.sorted,checked.visible);
   if([3,4,5].includes(face)){
    const png=await ev("(()=>{const v=window.nativeViewer;v.render();return v.renderer.domElement.toDataURL()})()");
    await writeFile(`${root}/source-${s+1}-face-${face}.png`,Buffer.from(png.split(',')[1],'base64'));
   }
   records.push({source:s+1,face,...checked});
  }
  console.log('Verified source',s+1,'all six directions');
 }
 for(const [a,b] of meta.midpoint_edges){
  const point=meta.centers[a].map((x,k)=>(x+meta.centers[b][k])/2);
  const state=await ev(`(async()=>{const v=window.roomViewer,p=${JSON.stringify(point)},delta=p.map((x,k)=>x-v.camera.position.toArray()[k]);v.camera.position.fromArray(p);v.controls.target.add({x:delta[0],y:delta[1],z:delta[2]});v.controls.update();return await window.flushNative()})()`);
  assert.equal(state.weight,0,'Midpoint should use fused baseline');
 }
 await ev('window.inspectNative(0,3)');
 const before=await ev('window.roomViewer.camera.position.toArray()');
 await call('Input.dispatchMouseEvent',{type:'mousePressed',x:1100,y:700,button:'left',buttons:1,clickCount:1});
 await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:1180,y:620,button:'left',buttons:1});
 await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:1180,y:620,button:'left',buttons:0,clickCount:1});
 const after=await ev('window.roomViewer.camera.position.toArray()'),dragError=Math.hypot(...before.map((x,k)=>x-after[k]));assert(dragError<1e-5);
 assert.equal(errors.length,0,JSON.stringify(errors));
 await writeFile(`${root}/check.json`,JSON.stringify({passed:true,sources:meta.stations.length,directions:records.length,records,midpoints:meta.midpoint_edges.length,dragError,originalFilesByteIdentical:meta.original_files_byte_identical,errors},null,2));
 console.log('Native source checks passed',records.length,'directions');
}finally{ws.close()}
