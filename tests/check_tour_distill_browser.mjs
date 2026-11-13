import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
const scene=process.argv[2]||'tour-distilled-six-v1',root='four-room/scenes/'+scene+'/browser';await mkdir(root,{recursive:true});
const data=JSON.parse(await readFile('four-room/scenes/'+scene+'/data/dataset.json','utf8'));
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),errors=[],plyRequests=new Set();
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.method==='Network.requestWillBeSent'&&m.params.request.url.includes('.ply'))plyRequests.add(m.params.request.url);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 const screenSize=process.argv.includes('--hd')?1280:640;
 await call('Runtime.enable');await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true});await call('Emulation.setDeviceMetricsOverride',{width:screenSize,height:screenSize,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/scene-room.html?scene='+scene+'&inspect=3'});await call('Page.bringToFront');
 let ready=false;for(let i=0;i<180;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.roomViewer&&!!window.roomSortRefresh')){ready=true;break}}assert(ready);console.log('Single PLY loaded');
 const ids=data.views.map((v,i)=>({v,i,fov:90})).filter(({v})=>(v.kind==='anchor'&&(v.source===0||v.source===3)&&/-(3|4|5)$/.test(v.name))||v.name==='test-edge-1-2-0.47-3');
 if(process.argv.includes('--detail'))ids.push(...ids.filter(x=>x.v.kind==='anchor').map(x=>({...x,fov:50})));
 for(const {v,i,fov} of ids){
  const name=v.name+(fov===90?'':'-fov'+fov);
  await ev(`(async()=>{window.inspectRoom(${i});const v=window.roomViewer;v.camera.fov=${fov};v.camera.updateProjectionMatrix();window.roomSortRefresh.request();await window.roomSortRefresh.flush();v.update();v.render()})()`);
  const frame=await ev('(()=>{const v=window.roomViewer,c=v.camera;v.render();return {png:v.renderer.domElement.toDataURL(),position:c.position.toArray(),worldInverse:c.matrixWorldInverse.toArray(),projection:c.projectionMatrix.toArray(),width:v.renderer.domElement.width,height:v.renderer.domElement.height,degree:v.splatMesh.material.uniforms.sphericalHarmonicsDegree.value}})()');
  assert.equal(frame.degree,2);assert(Math.hypot(...frame.position.map((x,k)=>x-v.center[k]))<1e-5);
  await writeFile(root+'/'+name+'.png',Buffer.from(frame.png.split(',')[1],'base64'));delete frame.png;await writeFile(root+'/'+name+'.json',JSON.stringify(frame));console.log('Captured',name);
 }
 assert.equal(plyRequests.size,1,JSON.stringify([...plyRequests]));assert([...plyRequests][0].endsWith('/'+scene+'/final/fused.ply'));assert.equal(errors.length,0,JSON.stringify(errors));
 await writeFile(root+'/check.json',JSON.stringify({passed:true,views:ids.map(x=>x.v.name+(x.fov===90?'':'-fov'+x.fov)),plyRequests:[...plyRequests],errors},null,2));console.log('Independent model browser checks passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
