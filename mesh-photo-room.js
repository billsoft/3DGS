import * as T from 'three';
import {OrbitControls} from './mesh-orbit-controls.js';
import {installRoomLook,setRoomLook} from './room-look.js';
import {installRoomNavigation} from './room-navigation.js';
const id=new URLSearchParams(location.search).get('scene')||'mesh-photo-six-v4';
if(!/^[a-zA-Z0-9_-]+$/.test(id))throw Error('Invalid scene');
const base='/four-room/scenes/'+id+'/',status=document.querySelector('#status');
try{
 const get=async name=>{const r=await fetch(base+name);if(!r.ok)throw Error(name+' 无法读取');return r};
 const [meta,buffer]=await Promise.all([get('mesh.json').then(r=>r.json()),get('mesh.bin').then(r=>r.arrayBuffer())]);
 const rows=new Float32Array(buffer),count=rows.length/5;
 if(count!==meta.triangles*3)throw Error('网格数据长度错误');
 const position=new Float32Array(count*3),uv=new Float32Array(count*2);
 for(let i=0;i<count;i++){position.set(rows.subarray(i*5,i*5+3),i*3);uv[i*2]=rows[i*5+3];uv[i*2+1]=1-rows[i*5+4]}
 const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.BufferAttribute(position,3));geometry.setAttribute('uv',new T.BufferAttribute(uv,2));geometry.computeVertexNormals();
 const renderer=new T.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setSize(innerWidth,innerHeight);renderer.outputColorSpace=T.SRGBColorSpace;document.body.append(renderer.domElement);
 const scene=new T.Scene();scene.background=new T.Color(0x1b2026);
 const camera=new T.PerspectiveCamera(75,innerWidth/innerHeight,.015,250);camera.up.fromArray(meta.up);camera.position.fromArray(meta.centers[0]);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enablePan=false;controls.enableZoom=false;
 const view=(s,face)=>meta.views.find(v=>v.source===s&&v.name.endsWith('-'+face));
 const first=view(0,3);controls.target.copy(camera.position).add(new T.Vector3(...first.basis.map(r=>r[2])));controls.update();
 const loader=new T.TextureLoader();
 const materials=await Promise.all(meta.groups.map(async(g,i)=>{
  const texture=await loader.loadAsync(base+g.texture);texture.colorSpace=T.SRGBColorSpace;texture.wrapS=T.RepeatWrapping;texture.wrapT=T.ClampToEdgeWrapping;texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());geometry.addGroup(g.start,g.count,i);
  return new T.MeshBasicMaterial({map:texture,side:T.DoubleSide});
 }));
 const ownerMaterials=meta.groups.map(g=>new T.MeshBasicMaterial({color:new T.Color().setHSL((g.source*.61803398875)%1,.65,.55),side:T.DoubleSide}));
 const normalMaterial=new T.MeshNormalMaterial({side:T.DoubleSide}),wireMaterial=new T.MeshBasicMaterial({color:0xb5d7e9,wireframe:true,side:T.DoubleSide});
 const mesh=new T.Mesh(geometry,materials);scene.add(mesh);
 const render=()=>renderer.render(scene,camera);let moving=null;
 const cancel=()=>{moving=null};
 installRoomLook({camera,controls,element:renderer.domElement,render,onStart:cancel});installRoomNavigation({camera,controls,render,onMove:cancel});
 const nearest=()=>meta.centers.map((p,i)=>({i,d:camera.position.distanceTo(new T.Vector3(...p))})).sort((a,b)=>a.d-b.d)[0].i;
 const look=face=>{const v=view(nearest(),face);setRoomLook(camera,controls,new T.Vector3(...v.basis.map(r=>r[2])));render()};
 const inspect=(source,face=3)=>{cancel();camera.position.fromArray(meta.centers[source]);const v=view(source,face);setRoomLook(camera,controls,new T.Vector3(...v.basis.map(r=>r[2])));render();return camera.position.toArray()};
 function button(container,label,point){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{moving={from:camera.position.clone(),to:new T.Vector3(...point),start:performance.now()}};document.querySelector(container).append(b)}
 meta.centers.forEach((p,i)=>button('#stations',meta.stations[i]+'号机位',p));
 meta.midpoint_edges.forEach(([a,b])=>button('#midpoints',meta.stations[a]+'–'+meta.stations[b]+' 中点',meta.centers[a].map((v,k)=>(v+meta.centers[b][k])/2)));
 document.querySelector('#up').onclick=()=>look(4);document.querySelector('#down').onclick=()=>look(5);document.querySelector('#front').onclick=()=>look(3);
 function mode(value){mesh.material=value==='photo'?materials:value==='owner'?ownerMaterials:value==='solid'?normalMaterial:wireMaterial;render()}
 document.querySelector('#mode').onchange=e=>{mode(e.target.value);e.target.blur()};document.querySelector('#download').href=base+'textured-room.zip';
 addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);render()});
 function frame(now){if(moving){const t=Math.min(1,(now-moving.start)/1000),e=t*t*(3-2*t),p=moving.from.clone().lerp(moving.to,e),delta=p.clone().sub(camera.position);camera.position.copy(p);controls.target.add(delta);controls.update();render();if(t===1)moving=null}requestAnimationFrame(frame)}requestAnimationFrame(frame);
 inspect(0);status.textContent='已加载 · '+meta.stations.length+' 张原片 · '+meta.triangles.toLocaleString()+' 个三角面';
 window.meshViewer={camera,controls,renderer,scene,mesh,meta,inspect,mode,render};window.meshReady=true;
}catch(error){status.textContent='加载失败：'+error.message;console.error(error)}
