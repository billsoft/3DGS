// Compare multi-source playback with an independently loaded single raw PLY.
// Run after check_native_scene.mjs in the same task-owned browser.
import {writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=process.argv[2]||'four-room/scenes/native-source-six-v1/standalone';
await mkdir(root,{recursive:true});
const tabs=await(await fetch('http://localhost:19346/json/list')).json();
const tab=tabs.find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
assert(tab);const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method))},180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 const meta=await ev('window.nativeMeta');
 await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 for(let s=0;s<meta.stations.length;s++){
  await call('Page.navigate',{url:'http://127.0.0.1:7870/native-scene-room.html?scene=native-source-six-v1&standalone=1&source='+s});
  let ready=false;for(let i=0;i<90;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.nativeState?.ready')){ready=true;break}}assert(ready);
  await ev('window.roomViewer.camera.fov=90;window.roomViewer.camera.updateProjectionMatrix()');
  for(const face of [3,4,5]){
   const state=await ev(`window.inspectNative(${s},${face})`);assert.equal(state.source,s);assert.equal(state.weight,1);
   assert.equal(await ev('window.nativeViewer.splatMesh.scenes.length'),1);
   const png=await ev('(()=>{const v=window.nativeViewer;v.render();return v.renderer.domElement.toDataURL()})()');
   await writeFile(`${root}/source-${s+1}-face-${face}.png`,Buffer.from(png.split(',')[1],'base64'));
  }
  console.log('Single original reference',s+1,'captured');
 }
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
