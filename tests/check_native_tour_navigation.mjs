import {mkdir,writeFile} from 'node:fs/promises';import assert from 'node:assert/strict';
const root='four-room/scenes/native-source-six-v1/tour-browser';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),600000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 await call('Runtime.enable');await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-tour-room.html?scene=native-source-six-v1&qa=settled'});
 await call('Page.bringToFront');await call('Emulation.setFocusEmulationEnabled',{enabled:true});
 let ready=false;for(let i=0;i<120;i++){await pause(1000);if(await ev('!!window.tourState?.ready')){ready=true;break}}assert(ready);console.log('Tour ready');
 await ev('window.roomViewer.camera.fov=90;window.roomViewer.camera.updateProjectionMatrix()');
 const meta=await ev('window.tourMeta'),p=meta.centers[0].map((x,k)=>(x+meta.centers[1][k])/2);
 await ev(`(async()=>{document.querySelector('#rest').checked=false;document.querySelector('#rest').dispatchEvent(new Event('change'));const v=window.roomViewer,old=v.camera.position.clone();v.camera.position.fromArray(${JSON.stringify(p)});v.controls.target.add(v.camera.position.clone().sub(old));v.controls.update();return await window.flushTour()})()`);
 const blend=await ev('window.tourState');assert(Math.abs(blend.weights[0]-.5)<1e-6);assert(Math.abs(blend.weights[1]-.5)<1e-6);console.log('Midpoint blend ready');
 await ev("document.querySelector('#rest').checked=true;document.querySelector('#rest').dispatchEvent(new Event('change'))");
 let settled;for(let i=0;i<60;i++){await pause(500);await call('Page.captureScreenshot',{format:'png'});settled=await ev('window.tourState');if(settled.settled&&settled.active.length===1)break}
 assert.equal(settled.active.length,1);assert.equal(settled.weights[settled.source],1);assert(Math.hypot(...settled.worldPosition.map((x,k)=>x-p[k]))<1e-5);
 await ev('window.flushTour()');await ev('document.querySelector("#tools").style.visibility="hidden"');
 const shot=await call('Page.captureScreenshot',{format:'png'});await writeFile(root+'/midpoint-settled.png',Buffer.from(shot.data,'base64'));await ev('document.querySelector("#tools").style.visibility="visible"');console.log('Midpoint settled to one original source without eye movement');
 await ev('window.inspectTour(0,3)');
 const before=await ev('window.roomViewer.camera.position.toArray()'),up=meta.up;
 await ev('document.activeElement?.blur()');
 await call('Input.dispatchKeyEvent',{type:'keyDown',key:'w',code:'KeyW',windowsVirtualKeyCode:87});
 try{for(let i=0;i<20;i++){await pause(500);await call('Page.captureScreenshot',{format:'png'});const p=await ev('window.roomViewer.camera.position.toArray()');if(Math.hypot(...p.map((x,k)=>x-before[k]))>.01)break}}finally{await call('Input.dispatchKeyEvent',{type:'keyUp',key:'w',code:'KeyW',windowsVirtualKeyCode:87})}
 const after=await ev('window.roomViewer.camera.position.toArray()'),delta=after.map((x,k)=>x-before[k]);assert(Math.hypot(...delta)>.01);assert(Math.abs(delta.reduce((s,x,k)=>s+x*up[k],0))<1e-5);console.log('W moves horizontally');
 await writeFile(root+'/navigation-checkpoint.json',JSON.stringify({midpointBlend:blend.weights,midpointSettled:settled.weights,keyboardDelta:delta},null,2));
 await ev('window.inspectTour(0,3)');
 await ev('window.tourTrace=[];window.tourTraceTimer=setInterval(()=>window.tourTrace.push({position:window.roomViewer.camera.position.toArray(),route:window.tourState?.travel?.route}),100)');
 await ev('window.goTour(3)');await ev('clearInterval(window.tourTraceTimer)');const trace=await ev('window.tourTrace'),arrival=await ev('window.tourState');assert.equal(arrival.weights[3],1);
 const route=trace.find(x=>x.route)?.route;assert(route&&route.length>=3);for(let i=1;i<route.length;i++)assert(meta.midpoint_edges.some(([a,b])=>(a===route[i-1]&&b===route[i])||(b===route[i-1]&&a===route[i])));
 console.log('Multi-edge route arrived',route);
 await ev('window.tourCancelledDone=false;window.goTour(0).then(()=>window.tourCancelledDone=true)');await pause(300);
 await call('Input.dispatchMouseEvent',{type:'mousePressed',x:1050,y:650,button:'left',buttons:1,clickCount:1});await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:1050,y:650,button:'left',buttons:0,clickCount:1});
 await pause(1000);assert.equal(await ev('window.tourState.travel'),null);
 const stopped=await ev('window.roomViewer.camera.position.toArray()');await pause(1000);const still=await ev('window.roomViewer.camera.position.toArray()');assert(Math.hypot(...still.map((x,k)=>x-stopped[k]))<1e-5);
 assert.equal(errors.length,0,JSON.stringify(errors));
 await writeFile(root+'/navigation.json',JSON.stringify({passed:true,midpointBlend:blend.weights,midpointSettled:settled.weights,keyboardDelta:delta,route,traceSamples:trace.length,cancelled:true,errors},null,2));console.log('Expanded tour navigation checks passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
