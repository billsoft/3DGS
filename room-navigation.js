import {Vector3} from './three.js';

// Translate camera and orbit target together. W/S follow the floor plane,
// independently of pitch; Q/E follow the scene's calibrated vertical axis.
export function installRoomNavigation({camera,controls,render=()=>{},onMove=()=>{},speed=1.2}) {
    const keys=new Set(), codes=new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE']);
    const up=camera.up.clone().normalize(), forward=new Vector3(), right=new Vector3(), delta=new Vector3();
    const lastForward=controls.target.clone().sub(camera.position).projectOnPlane(up).normalize();
    const editable=e=>e.target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName||'');
    function down(e) {
        if(!codes.has(e.code)||editable(e)||e.ctrlKey||e.altKey||e.metaKey)return;
        e.preventDefault();e.stopImmediatePropagation();
        if(!keys.has(e.code)){keys.add(e.code);onMove();controls.dispatchEvent({type:'start'});}
    }
    function release(e) {
        if(!codes.has(e.code))return;
        const had=keys.delete(e.code);
        if(had){e.preventDefault();e.stopImmediatePropagation();}
    }
    const clear=()=>keys.clear();
    window.addEventListener('keydown',down,true);window.addEventListener('keyup',release,true);
    window.addEventListener('blur',clear);document.addEventListener('visibilitychange',clear);
    let previous=performance.now(),alive=true;
    function frame(now) {
        if(!alive)return;
        const dt=Math.min(.05,Math.max(0,(now-previous)/1000));previous=now;
        if(keys.size){
            camera.getWorldDirection(forward);forward.projectOnPlane(up);
            if(forward.lengthSq()>1e-8)lastForward.copy(forward.normalize());
            forward.copy(lastForward);right.crossVectors(forward,up).normalize();
            delta.set(0,0,0).addScaledVector(forward,Number(keys.has('KeyW'))-Number(keys.has('KeyS')))
                .addScaledVector(right,Number(keys.has('KeyD'))-Number(keys.has('KeyA')))
                .addScaledVector(up,Number(keys.has('KeyQ'))-Number(keys.has('KeyE')));
            if(delta.lengthSq()>0){delta.normalize().multiplyScalar(speed*dt);camera.position.add(delta);controls.target.add(delta);controls.update();render();}
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    return ()=>{alive=false;clear();window.removeEventListener('keydown',down,true);window.removeEventListener('keyup',release,true);window.removeEventListener('blur',clear);document.removeEventListener('visibilitychange',clear);};
}
