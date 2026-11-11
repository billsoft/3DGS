import assert from 'node:assert/strict';
import {anchorPolicy} from '../native-anchor-policy.js';
const centers=[[0,0,0],[3,0,0],[3,3,0],[0,3,0]];
centers.forEach((c,i)=>{const p=anchorPolicy(c,centers);assert.equal(p.source,i);assert.equal(p.weight,1);});
assert.equal(anchorPolicy([1.5,0,0],centers).weight,0);
let previous=1;
for(let x=0;x<=1.5;x+=.01){const w=anchorPolicy([x,0,0],centers).weight;assert.ok(w>=0&&w<=1);assert.ok(w<=previous+1e-10);previous=w;}
console.log('Native anchor endpoints, midpoint fallback and continuous fade passed');
