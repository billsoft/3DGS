import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const scene=process.argv[2]||'tour-distilled-six-v3',root=`four-room/scenes/${scene}/display-check`;
assert(/^[a-z0-9-]+$/.test(scene));await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));assert(tab);
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const timeout=setTimeout(()=>{ws.close();throw Error('Display test timeout')},240000),records=[];
try{
 await call('Emulation.setDeviceMetricsOverride',{width:800,height:600,deviceScaleFactor:2,mobile:false});
 for(const quality of ['standard','hd']){
  await call('Page.navigate',{url:`http://127.0.0.1:7870/scene-room.html?scene=${scene}&inspect=3&quality=${quality}`});
  let ready=false;for(let i=0;i<100;i++){await new Promise(r=>setTimeout(r,500));if(await ev('!!window.roomViewer&&!!window.roomSortRefresh')){ready=true;break}}assert(ready);
  await ev('window.roomViewer.render()');
  const size=await ev('({w:window.roomViewer.renderer.domElement.width,h:window.roomViewer.renderer.domElement.height,dpr:devicePixelRatio})');
  assert.equal(size.w,quality==='hd'?1600:800);assert.equal(size.h,quality==='hd'?1200:600);
  await ev('document.querySelector("#tools").style.visibility="hidden"');
  const png=await ev('window.roomViewer.renderer.domElement.toDataURL()');
  await writeFile(`${root}/${quality}.png`,Buffer.from(png.split(',')[1],'base64'));records.push({quality,...size});
 }
 await writeFile(`${root}/check.json`,JSON.stringify({passed:true,records},null,2));console.log(records);
}finally{clearTimeout(timeout);ws.close()}
