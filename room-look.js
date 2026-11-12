import {Vector3} from './three.js';

// OrbitControls moves the eye around its target. Room inspection instead keeps
// the optical center fixed and rotates the look direction about that center.
export function setRoomLook(camera,controls,direction) {
    if(direction.lengthSq()<1e-12)throw Error('Look direction must be nonzero');
    const origin=camera.position.clone();
    const distance=Math.max(1e-3,controls.target.distanceTo(origin));
    controls.target.copy(origin).addScaledVector(direction.clone().normalize(),distance);
    controls.update();
    const correction=origin.clone().sub(camera.position);
    controls.target.add(correction);camera.position.copy(origin);
    camera.lookAt(controls.target);camera.updateMatrixWorld(true);
}

export function installRoomLook({camera,controls,element,render=()=>{},onStart=()=>{},sensitivity=.003}) {
    controls.minPolarAngle=0;controls.maxPolarAngle=Math.PI;
    controls.enableDamping=false;controls.enableRotate=false;controls.autoRotate=false;
    const up=camera.up.clone().normalize();let pointer=null,lastX=0,lastY=0;
    const down=e=>{
        if(e.button!==0||pointer!==null)return;
        pointer=e.pointerId;lastX=e.clientX;lastY=e.clientY;
        element.setPointerCapture(pointer);e.preventDefault();e.stopImmediatePropagation();onStart();
        controls.dispatchEvent({type:'start'});
    };
    const move=e=>{
        if(e.pointerId!==pointer)return;
        const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
        const direction=camera.getWorldDirection(new Vector3()).applyAxisAngle(up,-dx*sensitivity);
        const right=new Vector3().crossVectors(direction,up).normalize();
        if(right.lengthSq()<1e-10)right.setFromMatrixColumn(camera.matrixWorld,0).projectOnPlane(up).normalize();
        const pitch=Math.asin(Math.max(-1,Math.min(1,direction.dot(up))));
        const nextPitch=Math.max(-Math.PI/2+.001,Math.min(Math.PI/2-.001,pitch-dy*sensitivity));
        direction.applyAxisAngle(right,nextPitch-pitch);
        setRoomLook(camera,controls,direction);render();e.preventDefault();e.stopImmediatePropagation();
    };
    const end=e=>{
        if(e.pointerId!==pointer)return;
        const id=pointer;pointer=null;if(element.hasPointerCapture(id))element.releasePointerCapture(id);
        controls.dispatchEvent({type:'end'});e.preventDefault();e.stopImmediatePropagation();
    };
    element.addEventListener('pointerdown',down,true);element.addEventListener('pointermove',move,true);
    for(const type of ['pointerup','pointercancel','lostpointercapture'])element.addEventListener(type,end,true);
    return ()=>{
        element.removeEventListener('pointerdown',down,true);element.removeEventListener('pointermove',move,true);
        for(const type of ['pointerup','pointercancel','lostpointercapture'])element.removeEventListener(type,end,true);
    };
}
