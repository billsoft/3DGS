// Run against the task's dedicated headless browser, after opening fusion-native-room.html.
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const tabs=await(await fetch('http://localhost:19346/json/list')).json();
const tab=tabs.find(t=>t.type==='page'&&t.url.endsWith('/fusion-native-room.html'));
if(!tab)throw Error('Open the native preview in the dedicated test browser first');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let sequence=0;const pending=new Map();
ws.onmessage=e=>{const x=JSON.parse(e.data);if(x.id){const p=pending.get(x.id);if(p){clearTimeout(p.timer);pending.delete(x.id);x.error?p.reject(Error(x.error.message)):p.resolve(x.result);}}};
const request=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},30000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
const evaluate=async expression=>{const r=await request('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(predicate){for(let i=0;i<60;i++){const state=await evaluate('window.nativeState');if(predicate(state))return state;await pause(1000);}throw Error('Camera policy did not settle');}
try {
 const meta=await evaluate("fetch('/four-room/fusion-native-fidelity/fusion.json').then(r=>r.json())");
 for(let i=0;i<4;i++){
  await evaluate(`window.nativeMode('auto');window.roomViewer.camera.position.fromArray(${JSON.stringify(meta.centers[i])});window.roomViewer.controls.target.set(4,0,-.6);window.roomViewer.controls.update();window.roomViewer.forceRenderNextFrame();`);
  const state=await waitFor(s=>s.baseReady&&s.source===i&&s.weight===1);assert.equal(state.baseReady,true);
 }
 const mid=meta.centers[0].map((x,k)=>(x+meta.centers[1][k])/2);
 await evaluate(`window.roomViewer.camera.position.fromArray(${JSON.stringify(mid)});window.roomViewer.controls.update();`);await waitFor(s=>s.weight===0);
 await evaluate(`window.roomViewer.camera.position.fromArray(${JSON.stringify(meta.centers[0])});window.roomViewer.controls.target.set(4,0,-.6);window.roomViewer.controls.update();document.querySelector('#tools').style.display='none';`);
 for(const mode of ['auto','baseline']){
  await evaluate(`window.nativeMode('${mode}');window.roomViewer.forceRenderNextFrame();window.nativeBaseViewer.forceRenderNextFrame();`);await waitFor(s=>s.weight===(mode==='auto'?1:0));await pause(2500);
  if(process.argv.includes('--screenshots')){
   const capture=await evaluate(`(()=>{const v=${mode==='auto'?'window.roomViewer':'window.nativeBaseViewer'};v.render();return v.renderer.domElement.toDataURL('image/png');})()`);
   await writeFile(`four-room/fusion-native-fidelity/browser-${mode}.png`,Buffer.from(capture.split(',')[1],'base64'));
  }
 }
 await evaluate("document.querySelector('#tools').style.display='';window.nativeMode('auto');");
 console.log('All four native endpoints, midpoint fallback and mode controls verified');
} finally {ws.close();}
