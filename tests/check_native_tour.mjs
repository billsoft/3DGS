import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root='four-room/scenes/native-source-six-v1/tour-browser';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
assert(tab);const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let sequence=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method))},180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-tour-room.html?scene=native-source-six-v1'});
 let ready=false;for(let i=0;i<120;i++){await pause(1000);if(await ev('!!window.tourState?.ready')){ready=true;break}}assert(ready,'Tour failed to load');
 const meta=await ev('window.tourMeta');await ev('window.roomViewer.camera.fov=90;window.roomViewer.camera.updateProjectionMatrix()');
 const records=[];
 for(let s=0;s<meta.stations.length;s++){
  for(const face of [3,4,5]){
   const state=await ev(`window.inspectTour(${s},${face})`);assert.equal(state.weights[s],1);assert.deepEqual(state.active,[s]);assert.equal(state.usesFusedModel,false);
   assert(Math.hypot(...state.worldPosition.map((x,k)=>x-meta.centers[s][k]))<1e-5);
   const png=await ev(`(()=>{const v=window.tourSlots.get(${s}).viewer;v.render();return v.renderer.domElement.toDataURL()})()`);
   await writeFile(`${root}/source-${s+1}-face-${face}.png`,Buffer.from(png.split(',')[1],'base64'));records.push({source:s,face,weights:state.weights});
  }
  console.log('Native tour endpoint',s+1,'verified');
 }
 await ev('window.inspectTour(0,3)');
 await ev('document.querySelector("#rest").checked=false;document.querySelector("#rest").dispatchEvent(new Event("change"))');
 const edge=meta.midpoint_edges.find(e=>e.includes(0)),a=0,b=edge.find(x=>x!==0),samples=[];
 for(const t of [.15,.3,.38,.44,.5,.56,.62,.7,.85,1]){
  const p=meta.centers[a].map((x,k)=>x*(1-t)+meta.centers[b][k]*t);
  const state=await ev(`(async()=>{const v=window.roomViewer,p=${JSON.stringify(p)},old=v.camera.position.clone();v.camera.position.fromArray(p);v.controls.target.add(v.camera.position.clone().sub(old));v.controls.update();return await window.flushTour()})()`);
  assert.equal(state.usesFusedModel,false);assert(Math.abs(state.weights.reduce((x,y)=>x+y,0)-1)<1e-8);
  if(t<=.38)assert(Math.abs(state.weights[a]-1)<1e-8);
  if(t===.5)assert(Math.abs(state.weights[a]-.5)<1e-6&&Math.abs(state.weights[b]-.5)<1e-6);
  samples.push({t,weights:state.weights});
  if([.3,.5,.7].includes(t)){
   await ev('document.querySelector("#tools").style.visibility="hidden"');
   const png=await call('Page.captureScreenshot',{format:'png'});await writeFile(`${root}/transition-${t}.png`,Buffer.from(png.data,'base64'));
   await ev('document.querySelector("#tools").style.visibility="visible"');
  }
 }
 await ev('document.querySelector("#rest").checked=true;document.querySelector("#rest").dispatchEvent(new Event("change"))');
 await ev('window.inspectTour(0,3)');
 await ev('window.goTour(1)');const arrival=await ev('window.tourState');assert.equal(arrival.weights[1],1);
 const before=await ev('window.roomViewer.camera.position.toArray()');
 await call('Input.dispatchMouseEvent',{type:'mousePressed',x:1050,y:650,button:'left',buttons:1,clickCount:1});
 await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:1150,y:580,button:'left',buttons:1});
 await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:1150,y:580,button:'left',buttons:0,clickCount:1});
 const after=await ev('window.roomViewer.camera.position.toArray()');assert(Math.hypot(...before.map((x,k)=>x-after[k]))<1e-5);
 assert.equal(errors.length,0,JSON.stringify(errors));
 await writeFile(root+'/check.json',JSON.stringify({passed:true,endpoints:records,transition:samples,actualTourArrival:arrival.source,rotationEyeError:Math.hypot(...before.map((x,k)=>x-after[k])),errors},null,2));console.log('Native tour browser checks passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
