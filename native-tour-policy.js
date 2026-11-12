// Image-level mixing of independently rendered native PLYs, never point alpha blending.
const distance=(a,b)=>Math.hypot(...a.map((x,k)=>x-b[k]));
const smooth=x=>{x=Math.max(0,Math.min(1,x));return x*x*(3-2*x)};
export function tourPolicy(position,centers,core=.38){
    if(!centers.length||core<0||core>=.5)throw Error('Invalid native tour configuration');
    const distances=centers.map(c=>distance(position,c));
    const order=distances.map((d,i)=>i).sort((a,b)=>distances[a]-distances[b]||a-b);
    const source=order[0],weights=centers.map(()=>0);weights[source]=1;
    if(order.length===1)return {source,weights,distances,neighbor:null,core};
    const neighbor=order[1],spacing=distance(centers[source],centers[neighbor]);
    if(spacing>1e-8){
        const gap=(distances[neighbor]-distances[source])/spacing;
        const other=.5*smooth(1-gap/(1-2*core));
        weights[source]=1-other;weights[neighbor]=other;
    }
    return {source,neighbor,weights,distances,core,spacing};
}
export function layerOpacities(weights){
    let total=0;return weights.map(w=>{total+=w;return w>0?w/total:0});
}
export function stationRoute(start,end,centers,edges){
    const cost=centers.map(()=>Infinity),previous=centers.map(()=>-1),visited=new Set();cost[start]=0;
    const neighbors=centers.map(()=>[]);
    for(const [a,b] of edges){neighbors[a].push(b);neighbors[b].push(a)}
    while(visited.size<centers.length){
        const u=cost.map((_,i)=>i).filter(i=>!visited.has(i)).sort((a,b)=>cost[a]-cost[b])[0];
        if(u===undefined||!Number.isFinite(cost[u]))break;if(u===end)break;visited.add(u);
        for(const v of neighbors[u]){const candidate=cost[u]+distance(centers[u],centers[v]);if(candidate<cost[v]){cost[v]=candidate;previous[v]=u}}
    }
    if(!Number.isFinite(cost[end]))throw Error('机位路线不连通');
    const route=[end];while(route[0]!==start)route.unshift(previous[route[0]]);return route;
}
