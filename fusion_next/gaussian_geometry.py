"""Geometric axes independent of the order of Gaussian scale parameters."""
import numpy as np
from scipy.spatial.transform import Rotation

def shortest_axis_normals(a):
    frame=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()
    return frame[np.arange(len(a)),:,np.argmin(a[:,7:10],axis=1)]
