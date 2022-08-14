import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import numpy as np


def toGLmat(camera_pose):
    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    camera_pose = yz_flip.dot(camera_pose.T)

    return camera_pose.T


def render_depth(mesh, camera_pose, intrinsic, size=(800, 600)):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    camera_pose = toGLmat(camera_pose)

    camera = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2],
                                       intrinsic[1, 2], znear=0.05, zfar=1000.0)

    scene.add(camera, pose=camera_pose, name='camera')
    light = pyrender.SpotLight(color=np.ones(3), intensity=20.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(size[0], size[1])
    color, depth = r.render(scene)
    r.delete()

    camera_node = list(scene.get_nodes(obj=camera))[0]

    scene.remove_node(camera_node)

    return color, depth
