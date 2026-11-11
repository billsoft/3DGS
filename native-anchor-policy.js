// Camera-local source fidelity. Supports are disjoint; no nearest-cloud centroids.
export function anchorPolicy(position, centers) {
  const distance = centers.map(c => Math.hypot(...c.map((x,k)=>x-position[k])));
  const source = distance.indexOf(Math.min(...distance));
  const spacing = Math.min(...centers.map((c,i)=>i===source?Infinity:Math.hypot(...c.map((x,k)=>x-centers[source][k]))));
  const inner = .08*spacing, outer = .24*spacing;
  const t = Math.max(0,Math.min(1,(distance[source]-inner)/(outer-inner)));
  return {source, weight:1-t*t*(3-2*t), distance:distance[source], inner, outer};
}
