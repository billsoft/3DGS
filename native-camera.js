import {Matrix4,Vector3,Quaternion} from './three.js';

export function nativeTransform(rows) {
    const world=new Matrix4().set(...rows.flat()),position=new Vector3(),rotation=new Quaternion(),scale=new Vector3();
    world.decompose(position,rotation,scale);
    if(scale.x<=0||Math.max(Math.abs(scale.x-scale.y),Math.abs(scale.x-scale.z))>1e-5)throw Error('Native camera requires a similarity transform');
    return {inverse:world.clone().invert(),rotation:rotation.invert(),scale:scale.x};
}

export function syncNativeCamera(worldCamera,localCamera,transform) {
    localCamera.position.copy(worldCamera.position).applyMatrix4(transform.inverse);
    localCamera.quaternion.copy(transform.rotation).multiply(worldCamera.quaternion);
    localCamera.up.copy(worldCamera.up).transformDirection(transform.inverse);
    localCamera.fov=worldCamera.fov;localCamera.aspect=worldCamera.aspect;
    localCamera.near=worldCamera.near/transform.scale;localCamera.far=worldCamera.far/transform.scale;
    localCamera.zoom=worldCamera.zoom;localCamera.updateProjectionMatrix();localCamera.updateMatrixWorld(true);
}
