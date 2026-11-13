import assert from 'node:assert/strict';
import {tourPolicy,layerOpacities,stationRoute,bridgeWeights} from '../native-tour-policy.js';
const centers=[[0,0,0],[3,0,0],[3,3,0],[0,3,0]],edges=[[0,1],[1,2],[2,3],[3,0]];
for(const n of [1,2,4,6,9]){const c=Array.from({length:n},(_,i)=>[i*3,0,0]);c.forEach((p,s)=>{const result=tourPolicy(p,c);assert.equal(result.weights[s],1);assert.equal(result.weights.reduce((a,b)=>a+b,0),1)})}
assert.equal(tourPolicy([1.1,0,0],centers).weights[0],1,'Pure source region must extend beyond the old .24 spacing cutoff');
assert.deepEqual(tourPolicy([1.5,0,0],centers).weights,[.5,.5,0,0]);
for(const core of [.30,.38,.44]){assert.equal(tourPolicy([3*core-1e-7,0,0],centers,core).weights[0],1);assert.equal(tourPolicy([1.5,0,0],centers,core).weights[0],.5)}
let previous=[1,0,0,0];for(let x=0;x<=3;x+=.001){const p=tourPolicy([x,0,0],centers);assert(Math.abs(p.weights.reduce((a,b)=>a+b,0)-1)<1e-12);assert(p.weights.every(w=>w>=0&&w<=1));assert(Math.max(...p.weights.map((w,i)=>Math.abs(w-previous[i])))<.003);previous=p.weights}
for(const weights of [[1,0,0],[.5,.5,0],[.2,.3,.5],[0,1,0]]){const alpha=layerOpacities(weights);let out=[0,0,0];alpha.forEach((a,i)=>{out=out.map((x,k)=>x*(1-a)+(k===i?a:0))});assert(out.every((x,k)=>Math.abs(x-weights[k])<1e-12),'Layer opacity must yield image weights without black leakage')}
assert.deepEqual(stationRoute(0,2,centers,edges),[0,1,2]);assert.throws(()=>stationRoute(0,2,centers,[[0,1]]));
console.log('Expanded domains, continuous pair weights, image composition, and graph routes passed');
for(const n of [2,4,6,9]){
 const c=Array.from({length:n},(_,i)=>[3*i,0,0]);
 for(let x=0;x<=3;x+=.001){const p=tourPolicy([x,0,0],c,.49),w=bridgeWeights(p.weights,true);assert(Math.abs(w.reduce((a,b)=>a+b,0)-1)<1e-10);if(x<1.47||x>1.53)assert.equal(w[n],0)}
 assert.equal(bridgeWeights(tourPolicy([1.5,0,0],c,.49).weights,true)[n],1);
 assert.equal(bridgeWeights(tourPolicy(c[0],c,.49).weights,true)[0],1);
 assert.equal(bridgeWeights(tourPolicy([1.5,0,0],c,.49).weights,false)[n],0);
 assert.equal(bridgeWeights(tourPolicy([1.5,0,0],c,.49).weights,true,true)[n],0);
}
console.log('N-camera narrow fused bridge, source fidelity and unavailable fallback passed');
