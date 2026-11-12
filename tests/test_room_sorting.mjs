import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// Browser module has no dependencies; import as a module without changing the
// repository's package type or executing any browser globals in Node.
const code=await readFile(new URL('../room-sorting.js',import.meta.url),'utf8');
const {finishRoomSort,installRoomSortRefresh}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
function mock(){
 const listeners=new Map();let active=0,peak=0,calls=0,rendered=0;
 const v={camera:{updateMatrixWorld(){}},controls:{addEventListener(k,f){listeners.set(k,f)},removeEventListener(k){listeners.delete(k)}},splatSortCount:0,splatRenderCount:10,sortPromise:null,
  async runSplatSort(force,all){assert(force&&all);calls++;active++;peak=Math.max(peak,active);v.sortPromise=new Promise(r=>setTimeout(()=>{active--;v.splatSortCount=10;r()},5))},
  forceRenderNextFrame(){assert.equal(active,0);rendered++}};
 return {v,listeners,stats:()=>({active,peak,calls,rendered})};
}
test('completion waits for worker, not just submission',async()=>{const m=mock();await finishRoomSort(m.v);assert.deepEqual(m.stats(),{active:0,peak:1,calls:1,rendered:1})});
test('rapid changes coalesce and a change during sorting gets a final refresh',async()=>{
 const m=mock(),r=installRoomSortRefresh(m.v,{delay:100});r.request();r.request();const p=r.flush();
 await new Promise(resolve=>setTimeout(resolve,1));r.request();await p;
 assert.equal(m.stats().calls,2);assert.equal(m.stats().peak,1);r.dispose();assert.equal(m.listeners.size,0);
});
test('dispose cancels pending refresh',async()=>{const m=mock(),r=installRoomSortRefresh(m.v,{delay:5});r.dispose();await new Promise(resolve=>setTimeout(resolve,15));assert.equal(m.stats().calls,0)});
