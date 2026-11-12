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
const settleSort=async()=>{const result=await evaluate("(async()=>{const v=window.roomViewer;if(window.roomSortRefresh){await window.roomSortRefresh.flush()}else{for(let i=0;i<5;i++){if(v.sortPromise)await v.sortPromise;v.camera.updateMatrixWorld(true);await v.runSplatSort(true,true);if(v.sortPromise)await v.sortPromise;if(v.splatSortCount===v.splatRenderCount)break}}v.forceRenderNextFrame();return {sorted:v.splatSortCount,visible:v.splatRenderCount}})()");assert.equal(result.sorted,result.visible,'Camera still has a partial splat sort')};
try{
 await request('Runtime.enable');
 await request('Emulation.setDeviceMetricsOverride',{width:640,height:480,deviceScaleFactor:1,mobile:false});
 if(!process.argv.includes('--reuse-page'))await request('Page.navigate',{url:process.env.PREVIEW_URL||'http://127.0.0.1:7870/refit-room.html'});
 let ready=false;
 for(let i=0;i<120;i++){await pause(1000);if(await evaluate('!!window.roomViewer')){ready=true;break}}
 assert(ready,'Preview did not load');
 console.log('Preview loaded; capturing actual WebGL output');
 assert.equal(errors.length,0,JSON.stringify(errors));
 assert.match(await evaluate("document.querySelector('#status').textContent"),/已加载/);
 await mkdir(root,{recursive:true});
 for(const name of (process.argv.includes('--sort-only')?[]:['room','lamp'])){
  if(name==='lamp')await evaluate('window.inspectRoom(window.sceneMeta?4:0)');
  await pause(2000);
  await settleSort();
  const data=await evaluate("(()=>{const v=window.roomViewer;v.render();return v.renderer.domElement.toDataURL('image/png')})()");
  await writeFile(`${root}/browser-${name}.png`,Buffer.from(data.split(',')[1],'base64'));
 }
 const result=await evaluate("({status:document.querySelector('#status').textContent,buttons:document.querySelectorAll('#positions button').length,download:document.querySelector('#download').getAttribute('href'),scenes:window.roomViewer.splatMesh.getSceneCount()})");
 const scene=await evaluate('window.sceneMeta||null');
 if(scene){
  const preview=new URL(process.env.PREVIEW_URL);const id=preview.searchParams.get('scene');
  const data=await(await fetch(preview.origin+'/four-room/scenes/'+id+'/data-512/dataset.json')).json();
  assert.equal(result.buttons,scene.stations.length+data.midpoint_edges.length+3);
  result.stationCount=scene.stations.length;result.labels=scene.station_labels||scene.stations;
  result.automaticSortRefresh=await evaluate('!!window.roomSortRefresh');
  result.shDegree=await evaluate('window.roomViewer.sphericalHarmonicsDegree');
  assert.equal(result.shDegree,scene.sh_degree||0);
  if(result.shDegree){
   result.loadedShDegree=await evaluate('window.roomViewer.splatMesh.minSphericalHarmonicsDegree');
   result.activeShDegree=await evaluate('window.roomViewer.splatMesh.material.uniforms.sphericalHarmonicsDegree.value');
   assert.equal(result.loadedShDegree,result.shDegree,'PLY lost its SH coefficients during browser loading');
   assert.equal(result.activeShDegree,result.shDegree,'Shader did not enable directional color');
   result.directionalColorChangesImage=await evaluate("(()=>{const v=window.roomViewer;v.setActiveSphericalHarmonicsDegrees(0);v.render();const dc=v.renderer.domElement.toDataURL();v.setActiveSphericalHarmonicsDegrees(2);v.render();return dc!==v.renderer.domElement.toDataURL()})()");
   assert(result.directionalColorChangesImage,'SH enabled but did not affect the framebuffer');
  }
  if(result.automaticSortRefresh){
   result.smallMoveResorted=await evaluate("(async()=>{const v=window.roomViewer;await window.roomSortRefresh.flush();const before=v.sortPromise;v.camera.position.x+=.02;v.controls.target.x+=.02;v.controls.update();await window.roomSortRefresh.flush();return before!==v.sortPromise&&v.splatSortCount===v.splatRenderCount})()");
   assert(result.smallMoveResorted,'A 0.02-unit camera movement did not refresh the sort');
  }
  if(process.argv.includes('--check-look-in-place')){
   result.lookStayedAtStation=await evaluate("(async()=>{const v=window.roomViewer,c=window.sceneMeta.centers.at(-1);v.camera.position.fromArray(c);v.controls.target.fromArray(c.map((x,k)=>x+(k===0?1:0)));v.controls.update();const before=v.camera.position.clone();[...document.querySelectorAll('#positions button')].find(b=>b.textContent==='向上检查').click();await window.roomSortRefresh.flush();return before.distanceTo(v.camera.position)<1e-5})()");
   assert(result.lookStayedAtStation,'Look-up teleported the camera away from its capture station');
   const before=await evaluate('({position:window.roomViewer.camera.position.toArray(),direction:window.roomViewer.controls.target.clone().sub(window.roomViewer.camera.position).normalize().toArray()})');
   await request('Input.dispatchMouseEvent',{type:'mousePressed',x:560,y:400,button:'left',buttons:1,clickCount:1});
   await request('Input.dispatchMouseEvent',{type:'mouseMoved',x:600,y:365,button:'left',buttons:1});
   await request('Input.dispatchMouseEvent',{type:'mouseReleased',x:600,y:365,button:'left',buttons:0,clickCount:1});
   const after=await evaluate('({position:window.roomViewer.camera.position.toArray(),direction:window.roomViewer.controls.target.clone().sub(window.roomViewer.camera.position).normalize().toArray()})');
   result.dragPositionError=Math.hypot(...before.position.map((x,k)=>x-after.position[k]));
   result.dragChangedDirection=Math.hypot(...before.direction.map((x,k)=>x-after.direction[k]))>1e-4;
   assert(result.dragPositionError<1e-5&&result.dragChangedDirection,'Dragging must rotate the view without orbiting the camera');
  }
  if(process.argv.includes('--diagnostic-poses')){
   const C=scene.centers,poses=[['source',C[0],data.views[3].basis],['midpoint',C[0].map((x,k)=>(x+C.at(-1)[k])/2),data.views[3].basis],['up',C[0],data.views[4].basis],['down',C[0],data.views[5].basis]];
   for(const [name,center,basis] of poses){
    await evaluate(`(()=>{const v=window.roomViewer,c=${JSON.stringify(center)},b=${JSON.stringify(basis)};v.camera.position.fromArray(c);v.controls.target.fromArray(c.map((x,k)=>x+b[k][2]));v.controls.update();v.forceRenderNextFrame()})()`);
    await pause(2000);
    await settleSort();
    const image=await evaluate("(()=>{const v=window.roomViewer;v.render();return v.renderer.domElement.toDataURL('image/png')})()");
    await writeFile(`${root}/browser-${name}.png`,Buffer.from(image.split(',')[1],'base64'));
   }
  }
 }else assert(result.buttons>=8);
 assert.equal(result.scenes,1);
 await writeFile(`${root}/browser-check.json`,JSON.stringify({passed:true,...result,errors},null,2));
 console.log(JSON.stringify(result));
}finally{ws.close()}
