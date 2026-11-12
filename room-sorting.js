// A settled camera must not retain the previous view's culled/partial sort.
// Viewer.runSplatSort() resolves when work is submitted; sortPromise resolves
// when the worker actually returns the indexes for alpha compositing.
export async function finishRoomSort(viewer) {
    for(let i=0;i<5;i++){
        if(viewer.sortPromise)await viewer.sortPromise;
        viewer.camera.updateMatrixWorld(true);
        await viewer.runSplatSort(true,true);
        if(viewer.sortPromise)await viewer.sortPromise;
        if(viewer.splatSortCount===viewer.splatRenderCount)break;
    }
    viewer.forceRenderNextFrame();
}

export function installRoomSortRefresh(viewer,{delay=60,onError=console.error}={}) {
    let revision=0,completed=0,timer=null,running=null,disposed=false;
    const schedule=()=>{
        if(disposed||running||timer!==null)return;
        timer=setTimeout(()=>{timer=null;start()},delay);
    };
    const start=()=>{
        if(disposed||running||revision===completed)return running;
        const requested=revision;
        running=finishRoomSort(viewer).then(()=>{completed=requested}).catch(error=>{
            completed=requested;onError(error);
        }).finally(()=>{
            running=null;
            if(!disposed&&revision!==completed)schedule();
        });
        return running;
    };
    const request=()=>{if(!disposed){revision++;schedule()}};
    const flush=async()=>{
        if(disposed)return;
        if(timer!==null){clearTimeout(timer);timer=null}
        do {await(running||start())} while(!disposed&&revision!==completed);
    };
    viewer.controls.addEventListener('change',request);
    request();
    return {request,flush,dispose(){disposed=true;if(timer!==null)clearTimeout(timer);viewer.controls.removeEventListener('change',request)}};
}
