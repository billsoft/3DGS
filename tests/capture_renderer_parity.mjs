// Capture the actual browser camera, not a merely similar evaluation cube face.
import {writeFile,mkdir} from 'node:fs/promises';
const root='four-room/scenes/noise-ablation-v1/browser-parity';await mkdir(root,{recursive:true});
const tab=(await(await fetch('http://localhost:19346/json/list')).json()).find(t=>t.type==='page'&&t.url.startsWith('http://127.0.0.1:7870/'));
const ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);let seq=0;const pending=new Map();
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>reject(Error('CDP timeout '+method)),180000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
try{
 await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:7870/scene-room.html?scene=view-color-sh2-v1&inspect=4'});
 let ready=false;for(let i=0;i<150;i++){await new Promise(r=>setTimeout(r,1000));if(await ev('!!window.roomViewer&&!!window.roomSortRefresh')){ready=true;break}}if(!ready)throw Error('Not ready');
 for(const face of [4,5]){
  await ev(`(async()=>{const v=window.roomViewer;window.inspectRoom(${face});v.camera.fov=90;v.camera.updateProjectionMatrix();window.roomSortRefresh.request();await window.roomSortRefresh.flush();v.update();v.render()})()`);
  const result=await ev('(()=>{const v=window.roomViewer,c=v.camera;v.render();return {png:v.renderer.domElement.toDataURL(),worldInverse:c.matrixWorldInverse.toArray(),position:c.position.toArray(),projection:c.projectionMatrix.toArray(),width:v.renderer.domElement.width,height:v.renderer.domElement.height,shDegree:v.splatMesh.material.uniforms.sphericalHarmonicsDegree.value,sort:v.splatSortCount,visible:v.splatRenderCount}})()');
  await writeFile(`${root}/face-${face}.png`,Buffer.from(result.png.split(',')[1],'base64'));delete result.png;await writeFile(`${root}/face-${face}.json`,JSON.stringify(result,null,2));console.log('Browser parity captured',face);
 }
}finally{for(const p of pending.values())clearTimeout(p.timer);ws.close()}
