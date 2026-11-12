// Integration check against the task-owned headless browser, not a user's tab.
import {writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=process.argv[2]||'four-room/refit-coherent';
const port=process.env.CDP_PORT||'19346';
const tabs=await(await fetch(`http://localhost:${port}/json/list`)).json();
const tab=tabs.find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
assert(tab,'No task-owned preview tab');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let sequence=0;const pending=new Map();const errors=[];
ws.onmessage=e=>{const x=JSON.parse(e.data);if(x.method==='Runtime.exceptionThrown')errors.push(x.params.exceptionDetails);if(x.id){const p=pending.get(x.id);if(p){clearTimeout(p.timer);pending.delete(x.id);x.error?p.reject(Error(x.error.message)):p.resolve(x.result)}}};
const request=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method))},180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await request('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 await request('Runtime.enable');
 await request('Emulation.setDeviceMetricsOverride',{width:640,height:480,deviceScaleFactor:1,mobile:false});
 if(!process.argv.includes('--reuse-page'))await request('Page.navigate',{url:'http://127.0.0.1:7870/refit-room.html'});
 let ready=false;
 for(let i=0;i<120;i++){await pause(1000);if(await evaluate('!!window.roomViewer')){ready=true;break}}
 assert(ready,'Preview did not load');
 console.log('Preview loaded; capturing actual WebGL output');
 assert.equal(errors.length,0,JSON.stringify(errors));
 assert.match(await evaluate("document.querySelector('#status').textContent"),/已加载/);
 await mkdir(root,{recursive:true});
 for(const name of ['room','lamp']){
  if(name==='lamp')await evaluate('window.inspectRoom(0)');
  await pause(2000);
  const data=await evaluate("(()=>{const v=window.roomViewer;v.render();return v.renderer.domElement.toDataURL('image/png')})()");
  await writeFile(`${root}/browser-${name}.png`,Buffer.from(data.split(',')[1],'base64'));
 }
 const result=await evaluate("({status:document.querySelector('#status').textContent,buttons:document.querySelectorAll('#positions button').length,download:document.querySelector('#download').getAttribute('href'),scenes:window.roomViewer.splatMesh.getSceneCount()})");
 assert(result.buttons>=8);assert.equal(result.scenes,1);
 await writeFile(`${root}/browser-check.json`,JSON.stringify({passed:true,...result,errors},null,2));
 console.log(JSON.stringify(result));
}finally{ws.close()}
