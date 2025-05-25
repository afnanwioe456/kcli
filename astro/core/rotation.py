import numpy as np
import numpy.linalg as npl
from numba import njit


@njit
def ENU_rotation(r):
    norm_r = npl.norm(r)
    if norm_r == 0:
        raise ValueError()
    up = r / norm_r

    z_axis = np.array([0, 0, 1])
    north = z_axis - np.dot(z_axis, up) * up  # 去除 up 分量
    norm_north = npl.norm(north)
    if norm_north < 1e-8:  # 若在极区附近
        north = np.array([0, 1, 0])
        norm_north = npl.norm(north)
    north = north / norm_north

    east = np.cross(north, up)
    east = east / npl.norm(east)
    
    return np.vstack((north, east, up))

@njit
def TNW_rotation(r, v):
    T = v / npl.norm(v)  # T方向：速度矢量单位化
    h = np.cross(r, v)  # 轨道角动量
    N = h / npl.norm(h)  # N方向：角动量单位化
    W = np.cross(N, T)  # W方向：完成右手系
    return np.vstack((T, W, N))

@njit
def PQW_rotation_rv(r, v, GM):
    r_norm = npl.norm(r)
    h = np.cross(r, v)
    h_norm = npl.norm(h)
    W = h / h_norm
    e = np.cross(v, h) / GM - r / r_norm
    e_norm = npl.norm(e)
    P = e / e_norm
    Q = np.cross(W, P)
    return np.vstack((P, Q, W))

@njit
def PQW_rotation_coe(inc, raan, argp):
    R_raan = np.array([[np.cos(raan), np.sin(raan), 0.],
                    [-np.sin(raan), np.cos(raan), 0.],
                    [0., 0., 1.]], dtype=np.float64)
    R_inc = np.array([[1., 0., 0.],
                   [0., np.cos(inc), np.sin(inc)],
                   [0., -np.sin(inc), np.cos(inc)]], dtype=np.float64)
    R_argp = np.array([[np.cos(argp), np.sin(argp), 0.],
                    [-np.sin(argp), np.cos(argp), 0.],
                    [0., 0., 1.]], dtype=np.float64)
    return np.dot(np.dot(R_argp, R_inc), R_raan)
