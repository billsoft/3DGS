import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map(),requests=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout')),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true})).result.value;
try{
 await call('Runtime.enable');await call('Network.enable');
 await call('Page.navigate',{url:'http://127.0.0.1:7870/native-tour-room.html?scene=native-photo-six-v3'});
 let retired=false;for(let i=0;i<30;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('document.querySelector("#status")?.textContent.includes("已撤回")')){retired=true;break}}
 assert(retired);assert(!requests.some(u=>u.endsWith('.ply')));const baseline=await ev('document.querySelector("#previous").href');assert(baseline.endsWith('native-source-six-v1&range=0.49'));
 await call('Page.navigate',{url:baseline});let ready=false;
 for(let i=0;i<180;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.tourState?.ready')){ready=true;break}}
 assert(ready);const state=await ev('window.tourState');assert.equal(state.core,.49);assert.equal(state.weights[0],1);assert.equal(state.usesFusedModel,false);
 await writeFile('four-room/scenes/native-teacher-audit-v1/rollback-browser.json',JSON.stringify({passed:true,retiredBlocked:true,baseline,state},null,2));console.log('Retired trial blocked; original 49% baseline restored');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
