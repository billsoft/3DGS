"""Run in the isolated Open3D environment; no photos or GPU required."""
import sys,unittest
from pathlib import Path
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from single_gs_mesh import camera_intrinsics,check_tsdf_backend


@unittest.skipIf(o3d is None,'Run with requirements-mesh.txt in the isolated Open3D environment')
class MeshDepthTest(unittest.TestCase):
    def test_backend_plane(self):
        check_tsdf_backend(o3d)

    def test_translated_rotated_camera_preserves_world_plane(self):
        # Camera looks along world +X. A depth of 2 from X=1 is plane X=3.
        size=64;center=np.array([1.,.2,-.3])
        basis=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        ext=np.eye(4);ext[:3,:3]=basis.T;ext[:3,3]=-basis.T@center
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((size,size,3),'u1')),
            o3d.geometry.Image(np.full((size,size),2.,'f')),
            depth_scale=1.,depth_trunc=30.,convert_rgb_to_intensity=False)
        volume=o3d.pipelines.integration.ScalableTSDFVolume(.04,.12,o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        volume.integrate(rgbd,camera_intrinsics(o3d,size),ext)
        mesh=volume.extract_triangle_mesh();self.assertGreater(len(mesh.triangles),100)
        np.testing.assert_allclose(np.asarray(mesh.vertices)[:,0],3.,atol=.02)


if __name__=='__main__':unittest.main()
