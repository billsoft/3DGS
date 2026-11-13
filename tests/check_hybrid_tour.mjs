import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root='four-room/scenes/native-source-six-v1/hybrid-browser';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout')),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:900,height:700,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-tour-room.html?scene=native-source-six-v1&fused=tour-appearance-six-v1'});await call('Page.bringToFront');
 let ready=false;for(let i=0;i<180;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.tourState?.ready')){ready=true;break}}assert(ready);console.log('Hybrid ready');
 const states=[];
 for(const face of [3,4,5]){const s=await ev(`window.inspectTour(0,${face})`);assert.equal(s.weights[0],1);assert.equal(s.usesFusedModel,false);states.push(s);
  await ev('document.querySelector("#tools").style.visibility="hidden"');const png=await call('Page.captureScreenshot',{format:'png'});await writeFile(`${root}/source-1-${face}.png`,Buffer.from(png.data,'base64'));await ev('document.querySelector("#tools").style.visibility="visible"');console.log('Pure source 1 face',face);
 }
 const meta=await ev('window.tourMeta'),n=meta.stations.length;
 await ev('document.querySelector("#rest").checked=false;document.querySelector("#rest").dispatchEvent(new Event("change"))');
 for(const t of [.48,.5,.52]){const p=meta.centers[0].map((x,k)=>x*(1-t)+meta.centers[1][k]*t);
  const s=await ev(`(async()=>{const v=window.roomViewer,old=v.camera.position.clone();v.camera.position.fromArray(${JSON.stringify(p)});v.controls.target.add(v.camera.position.clone().sub(old));v.controls.update();return await window.flushTour()})()`);
  assert.equal(s.usesFusedModel,t===.5);if(t===.5)assert(Math.abs(s.weights[n]-1)<1e-6);states.push(s);console.log('Bridge sample',t,s.weights);
  if(t===.5){const png=await call('Page.captureScreenshot',{format:'png'});await writeFile(`${root}/midpoint.png`,Buffer.from(png.data,'base64'))}
 }
 await ev('document.querySelector("#rest").checked=true;document.querySelector("#rest").dispatchEvent(new Event("change"))');
 await new Promise(r=>setTimeout(r,300));const settled=await ev('window.flushTour()');assert.equal(settled.usesFusedModel,false);
 await ev('window.goTour(1)');const arrival=await ev('window.tourState');assert.equal(arrival.weights[1],1);assert.equal(arrival.usesFusedModel,false);
 const final=await ev('window.inspectTour(0,5)');assert.equal(final.weights[0],1);assert.equal(final.usesFusedModel,false);assert.equal(errors.length,0);
 await writeFile(root+'/check.json',JSON.stringify({passed:true,states,settled,arrival,errors},null,2));console.log('Hybrid endpoint, narrow bridge, rest and route checks passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
