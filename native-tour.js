import * as THREE from 'three';
import {Viewer,RenderMode,SceneRevealMode} from './gaussian.js';
import {nativeTransform,syncNativeCamera} from './native-camera.js';
import {tourPolicy,layerOpacities,stationRoute,bridgeWeights} from './native-tour-policy.js';
import {installRoomLook,setRoomLook} from './room-look.js';
import {installRoomNavigation} from './room-navigation.js';
import {installRoomSortRefresh} from './room-sorting.js';

const query=new URLSearchParams(location.search),id=query.get('scene')||'native-source-six-v1';
if(!/^[a-zA-Z0-9_-]+$/.test(id))throw Error('Invalid scene');
const root='/four-room/scenes/'+id,status=document.querySelector('#status'),detail=document.querySelector('#detail');
try{
 const response=await fetch(root+'/native.json');if(!response.ok)throw Error('场景信息无法读取');const meta=await response.json();
 const fusedId=query.get('fused'),hybrid=!!fusedId,FUSED=meta.stations.length;
 const sourceName=meta.photo_recolored?'原片附色':'原始';
 if(meta.photo_recolored){
  document.querySelector('h1').textContent='逐机位原片附色 · 连续漫游';
  document.querySelector('#intro').textContent='每份 PLY 只用对应全景图附色；几何与透明度保持不变。';
  document.querySelector('#single').textContent='下载当前机位 PLY';document.querySelector('#single').setAttribute('download','');
  document.querySelector('#range').parentElement.firstChild.textContent='机位范围 ';
  document.querySelector('#explanation').textContent='每个机位使用各自的模型，未通过附色验证的机位保留原始 PLY。移动中可能有双影。';
  const report=document.createElement('a');report.href=root+'/index.html';report.textContent='原片与附色对照';report.target='_blank';document.querySelector('#single').parentNode.append(report);
 }
 let fusedMeta=null,fusedPath=null;
 if(hybrid){
  if(!/^[a-zA-Z0-9_-]+$/.test(fusedId))throw Error('Invalid fused scene');
  const r=await fetch('/four-room/scenes/'+fusedId+'/fusion.json');if(!r.ok)throw Error('融合模型信息无法读取');fusedMeta=await r.json();
  if(fusedMeta.centers?.length!==meta.centers.length||meta.centers.some((c,i)=>Math.hypot(...c.map((v,k)=>v-fusedMeta.centers[i][k]))>1e-4))throw Error('原始与融合模型坐标不一致，不能直接切换');
  fusedPath='/four-room/scenes/'+fusedId+'/'+fusedMeta.model_folder+'/fused.ply';
  document.querySelector('h1').textContent=sourceName+'机位保真 · 融合模型过渡';
  document.querySelector('#intro').textContent='拍摄点附近使用对应机位 PLY；跨机位的短暂过渡使用融合 PLY。';
  document.querySelector('#explanation').textContent='移动到交接中心时使用融合模型；两侧渐入渐出。勾选清晰优先后，停下恢复附近机位的模型。';
 }
 const first=Math.max(0,Math.min(meta.stations.length-1,Number(query.get('source'))||0));
 const centers=meta.centers.map(c=>new THREE.Vector3(...c)),transforms=meta.stations.map(s=>nativeTransform(s.world_from_raw));
 if(hybrid)transforms.push(nativeTransform([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]));
 const viewFor=(source,face)=>meta.views.find(v=>v.source===source&&v.name.endsWith('-'+face));
 const initial=viewFor(first,3),target=initial.center.map((x,k)=>x+initial.basis[k][2]);
 const options={selfDrivenMode:false,cameraUp:meta.up,initialCameraPosition:meta.centers[first],initialCameraLookAt:target,sharedMemoryForWorkers:false,integerBasedSort:true,gpuAcceleratedSort:false,ignoreDevicePixelRatio:true,sphericalHarmonicsDegree:0,renderMode:RenderMode.OnChange,sceneRevealMode:SceneRevealMode.Instant};
 const control=new Viewer({...options,rootElement:document.querySelector('#control')});
 const slots=new Map(),weights=Array(meta.stations.length+(hybrid?1:0)).fill(0),buttons=[];
 let ready=false,alive=true,serial=Promise.resolve(),core=.38,revision=0,travel=null,lastTime=performance.now(),lastTrim=0,restClarity=true,lastTranslation=0;
 if(hybrid||meta.photo_recolored){core=.49;document.querySelector('#range').value=String(core)}
 const lastPosition=control.camera.position.clone();
 const force=()=>{for(const slot of slots.values())if(slot.ready)slot.viewer.forceRenderNextFrame()};
 const cancel=()=>{revision++;travel=null};
 const disposeLook=installRoomLook({camera:control.camera,controls:control.controls,element:control.renderer.domElement,render:force,onStart:cancel});
 const disposeNavigation=installRoomNavigation({camera:control.camera,controls:control.controls,render:force,onMove:cancel});
 control.controls.enablePan=false;
 control.controls.addEventListener('start',cancel);
 function policy(){return tourPolicy(control.camera.position.toArray(),meta.centers,core)}
 function ensure(source){
  if(slots.has(source)){const slot=slots.get(source);slot.used=performance.now();return slot.promise}
  const slot={source,ready:false,used:performance.now(),weight:0};slots.set(source,slot);
  slot.promise=serial.then(async()=>{
   if(!alive)return;
   const layer=document.createElement('div');layer.className='tour-layer';layer.style.zIndex=String(source+1);document.body.append(layer);slot.layer=layer;
   const viewer=new Viewer({...options,rootElement:layer,useBuiltInControls:false,sphericalHarmonicsDegree:source===FUSED?(fusedMeta.sh_degree||0):0});slot.viewer=viewer;
   await viewer.addSplatScenes([{path:source===FUSED?fusedPath:root+'/'+meta.stations[source].file,splatAlphaRemovalThreshold:0}],false);
   viewer.renderer.setClearColor(0,1);syncNativeCamera(control.camera,viewer.camera,transforms[source]);viewer.update();
   slot.sort=installRoomSortRefresh(viewer,{controls:control.controls});await slot.sort.flush();viewer.render();slot.ready=true;
  }).catch(error=>{slot.error=error;detail.textContent=(source===FUSED?'融合模型':'机位 '+meta.stations[source].label)+' 加载失败：'+error.message;throw error});
  serial=slot.promise.catch(()=>{});return slot.promise;
 }
 function preload(p){
  const wanted=[p.source,p.neighbor].filter(s=>s!==null&&s!==undefined);
  if(hybrid)wanted.push(FUSED);
  for(const s of wanted)ensure(s).catch(()=>{});
  return new Set(wanted);
 }
 function sync(slot){syncNativeCamera(control.camera,slot.viewer.camera,transforms[slot.source]);slot.viewer.update()}
 function desired(p,settled){
  const target=hybrid?bridgeWeights(p.weights,slots.get(FUSED)?.ready,settled):[...p.weights];
  if(settled){
   const current=weights.slice(0,FUSED).indexOf(Math.max(...weights.slice(0,FUSED)));
   const tied=current>=0&&p.spacing&&p.distances[current]-p.distances[p.source]<.02*p.spacing;
   const best=tied&&slots.get(current)?.ready?current:p.source;
   if(slots.get(best)?.ready){target.fill(0);target[best]=1}
  }
  const result=target.map((w,i)=>slots.get(i)?.ready?w:0),total=result.reduce((a,b)=>a+b,0);
  if(total>0)return result.map(w=>w/total);
  const existing=[...slots.values()].filter(s=>s.ready).sort((a,b)=>weights[b.source]-weights[a.source])[0];
  if(existing)result[existing.source]=1;return result;
 }
 function compose(dt=0,instant=false){
  control.updateForRendererSizeChanges();
  if(lastPosition.distanceToSquared(control.camera.position)>1e-12){lastTranslation=performance.now();lastPosition.copy(control.camera.position)}
  const settled=restClarity&&!travel&&performance.now()-lastTranslation>180;
  const p=policy(),wanted=preload(p),goal=desired(p,settled);
  const atCenter=p.distances[p.source]<1e-6&&slots.get(p.source)?.ready;
  const ease=instant||atCenter?1:1-Math.exp(-Math.min(dt,.5)/.12);
  for(let i=0;i<weights.length;i++){weights[i]+=(goal[i]-weights[i])*ease;if(weights[i]<.001&&goal[i]===0)weights[i]=0}
  if(hybrid&&(settled||p.neighbor===null||p.weights[p.neighbor]===0))weights[FUSED]=0;
  const total=weights.reduce((a,b)=>a+b,0);if(total>0)weights.forEach((w,i)=>weights[i]=w/total);
  const opacity=layerOpacities(weights),active=[];
  for(const [s,slot] of slots){
   if(!slot.ready)continue;
   if(weights[s]>0||wanted.has(s)){sync(slot);if(weights[s]>0&&slot.viewer.shouldRender())slot.viewer.render();slot.viewer.renderNextFrame=false}
   slot.layer.style.opacity=String(opacity[s]);slot.weight=weights[s];
   if(weights[s]>0){slot.used=performance.now();active.push(s)}
  }
  const dominant=weights.slice(0,FUSED).some(w=>w>0)?weights.slice(0,FUSED).indexOf(Math.max(...weights.slice(0,FUSED))):p.source;
  const selectedName=meta.stations[dominant].photo_refit_passed===false?'原始（附色未通过）':sourceName;
  status.textContent=active.length===1&&active[0]!==FUSED?selectedName+' '+meta.stations[dominant].label+' 号机位 · 单份模型':'过渡 · '+active.map(s=>(s===FUSED?'融合模型':meta.stations[s].label+'号')+' '+Math.round(weights[s]*100)+'%').join(' / ');
  const failed=slots.get(p.source)?.error||slots.get(p.neighbor)?.error||(hybrid&&slots.get(FUSED)?.error);
  detail.textContent=failed?'机位加载失败，保留已加载模型：'+failed.message:travel?'沿机位路线移动，拖动或按移动键可停止。':!slots.get(p.neighbor)?.ready&&p.neighbor!==null?'正在预载相邻机位…':'小范围自由移动；点击机位可体验看房式过渡。';
  buttons.forEach((b,i)=>b.classList.toggle('active',i===dominant));
  document.querySelector('#single').href=meta.photo_recolored?root+'/'+meta.stations[dominant].file:'/native-scene-room.html?scene='+id+'&standalone=1&source='+dominant;
  window.tourState={ready,weights:[...weights],source:dominant,active,core,settled,restClarity,worldPosition:control.camera.position.toArray(),policy:p,loaded:[...slots.values()].filter(s=>s.ready).map(s=>s.source),travel:travel?{route:travel.route,target:travel.target}:null,usesFusedModel:hybrid&&weights[FUSED]>0,fusedScene:fusedId};
  if(performance.now()-lastTrim>2000){lastTrim=performance.now();trim(wanted)}
  return window.tourState;
 }
 function trim(wanted){
  const unused=[...slots.values()].filter(s=>s.ready&&weights[s.source]===0&&!wanted.has(s.source)&&!travel?.remaining.includes(s.source)).sort((a,b)=>a.used-b.used);
  while(slots.size>4&&unused.length){const slot=unused.shift();slots.delete(slot.source);slot.sort.dispose();const renderer=slot.viewer.renderer;slot.viewer.dispose().then(()=>renderer.forceContextLoss()).catch(console.error)}
 }
 async function flush(instant=true){
  const p=policy();await Promise.all([...p.weights.map((w,s)=>w>0?ensure(s):Promise.resolve()),...(hybrid?[ensure(FUSED).catch(()=>{})]:[])]);compose(0,instant);
  for(const slot of slots.values())if(slot.ready&&weights[slot.source]>0){sync(slot);slot.sort.request();await slot.sort.flush();slot.viewer.render()}
  return compose(0,instant);
 }
 function translate(point){const delta=point.clone().sub(control.camera.position);control.camera.position.copy(point);control.controls.target.add(delta);control.controls.update();force()}
 async function inspect(source,face=3){cancel();await ensure(source);const v=viewFor(source,face);control.camera.position.fromArray(v.center);setRoomLook(control.camera,control.controls,new THREE.Vector3(...v.basis.map(r=>r[2])));return flush()}
 async function segment(point,token){
  const from=control.camera.position.clone(),length=from.distanceTo(point);if(length<1e-6)return true;
  const duration=Math.max(1000,Math.min(2600,length/1.5*1000));let elapsed=0,previous=performance.now();
  return new Promise(resolve=>{const step=now=>{if(token!==revision||!alive){resolve(false);return}elapsed+=Math.min(80,Math.max(0,now-previous));previous=now;const t=Math.min(1,elapsed/duration),e=t*t*t*(10-15*t+6*t*t);translate(from.clone().lerp(point,e));if(t<1)requestAnimationFrame(step);else resolve(true)};requestAnimationFrame(step)});
 }
 async function go(source){
  if(hybrid)await ensure(FUSED).catch(()=>{});
  cancel();const token=revision,start=policy().source,route=stationRoute(start,source,meta.centers,meta.midpoint_edges);travel={route,remaining:[...route],target:source};
  for(let i=0;i<route.length;i++){
   await ensure(route[i]);if(token!==revision)return;
   if(i+1<route.length)ensure(route[i+1]).catch(()=>{});
   if(!await segment(centers[route[i]],token))return;
   if(token===revision)travel.remaining=route.slice(i+1);
  }
  if(token===revision){travel=null;await flush()}
 }
 function look(face){cancel();const v=viewFor(policy().source,face);setRoomLook(control.camera,control.controls,new THREE.Vector3(...v.basis.map(r=>r[2])));force()}
 meta.stations.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.label+'号机位';b.onclick=()=>go(i).catch(e=>{travel=null;detail.textContent=e.message});document.querySelector('#stations').append(b);buttons.push(b)});
 document.querySelector('#range').onchange=e=>{core=Number(e.target.value);e.target.blur();force()};
 document.querySelector('#rest').onchange=e=>{restClarity=e.target.checked;e.target.blur();force()};
 document.querySelector('#down').onclick=()=>look(5);document.querySelector('#up').onclick=()=>look(4);document.querySelector('#front').onclick=()=>look(3);
 document.querySelector('#previous').href=(hybrid||meta.photo_recolored?'/native-tour-room.html':'/native-scene-room.html')+'?scene='+(meta.reference_native_scene||id);
 window.roomViewer=control;window.tourMeta=meta;window.tourSlots=slots;window.inspectTour=inspect;window.flushTour=flush;window.goTour=go;
 await ensure(first);weights[first]=1;ready=true;await inspect(first,3);
 function frame(now){if(!alive)return;compose((now-lastTime)/1000);lastTime=now;requestAnimationFrame(frame)}requestAnimationFrame(frame);
 window.addEventListener('pagehide',()=>{alive=false;cancel();disposeLook();disposeNavigation();for(const s of slots.values())s.sort?.dispose()},{once:true});
}catch(error){status.textContent='加载失败：'+error.message;console.error(error)}
