// Uses the task-owned development browser, never a user's daily browser.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const scene=process.argv[2];assert(/^[a-zA-Z0-9_-]+$/.test(scene));
const endpoint='http://localhost:19346';
const tab=await(await fetch(endpoint+'/json/new?'+encodeURIComponent('http://127.0.0.1:7870/tour-distill-report.html?scene='+scene),{method:'PUT'})).json();
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(p){pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout')),30000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 let state;
 for(let i=0;i<60;i++){
  state=await ev(`(()=>({status:document.querySelector('#status')?.textContent,tables:document.querySelectorAll('#summary table').length,buttons:document.querySelectorAll('#summary button:not(:disabled)').length,loaded:!!document.querySelector('#comparison')?.naturalWidth,links:[...document.querySelectorAll('a')].map(a=>a.href)}))()`);
  if(state.tables===2&&state.loaded)break;await new Promise(r=>setTimeout(r,500));
 }
 assert.equal(state.tables,2);assert(state.loaded);assert(state.buttons>0);
 assert(state.status.includes('尚未达到'));
 assert(state.links.some(x=>x.includes('tour-appearance-six-v1')));
 assert(state.links.some(x=>x.includes('native-source-six-v1&range=0.49')));
 await ev("document.querySelector('#summary button:not(:disabled)').click()");
 const selected=await ev("document.querySelector('#views').value");assert(selected);
 for(const url of state.links){const res=await fetch(url,{method:'HEAD'});assert.equal(res.status,200,url)}
 await writeFile('four-room/scenes/'+scene+'/browser/report-check.json',JSON.stringify({passed:true,...state,selectedRegression:selected},null,2));console.log('Report, old baseline, native range, download and local-regression links passed');
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close();await fetch(endpoint+'/json/close/'+tab.id)}
