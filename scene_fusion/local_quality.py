"""RGB/chroma patch checks so broad uniform surfaces cannot hide local errors."""
import numpy as np

def local_metrics(image,target,alpha,tile=32):
    error=np.asarray(image)-np.asarray(target);mask=np.asarray(alpha)>.95
    color=np.abs(error).mean(2)
    chroma=np.abs(error-error.mean(2,keepdims=True)).mean(2)
    h,w=mask.shape;values=[]
    for y in range(0,h-tile+1,tile):
        for x in range(0,w-tile+1,tile):
            m=mask[y:y+tile,x:x+tile]
            if m.mean()>.8:values.append(float(color[y:y+tile,x:x+tile][m].mean()))
    return dict(chroma_mae=float(chroma[mask].mean()) if mask.any() else None,
                patch_p95=float(np.percentile(values,95)) if values else None,
                patch_max=max(values) if values else None,valid_patches=len(values))
