import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const scene=process.argv[2]||'auto-registration-six-v1';
assert(/^[a-z0-9-]+$/.test(scene));
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
assert(tab,'Expected task-owned development tab');
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const timeout=setTimeout(()=>{ws.close();throw Error('Report check timed out')},30000);
try{
 await call('Emulation.setDeviceMetricsOverride',{width:1280,height:1100,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:`http://127.0.0.1:7870/four-room/scenes/${scene}/comparison/index.html`});
 for(let i=0;i<30;i++){
  await new Promise(r=>setTimeout(r,200));
  const r=await call('Runtime.evaluate',{expression:'document.querySelectorAll("table tr").length',returnByValue:true});
  if(r.result.value>1)break;
 }
 const result=await call('Runtime.evaluate',{expression:'({rows:document.querySelectorAll("table tr").length,markers:document.querySelectorAll("svg circle").length,width:document.documentElement.scrollWidth,viewport:innerWidth,text:document.body.innerText})',returnByValue:true});
 const data=result.result.value;assert.equal(data.markers,2*(data.rows-1));assert(data.rows>2);assert(data.width<=data.viewport);assert(data.text.includes('不能认定全面优于旧版'));
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(`four-room/scenes/${scene}/comparison/browser.png`,Buffer.from(shot.data,'base64'));
 console.log(JSON.stringify({passed:true,rows:data.rows,markers:data.markers,overflow:false}));
}finally{clearTimeout(timeout);ws.close()}
