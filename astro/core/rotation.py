from numpy import cos, sin, dot, cross, float64, array, vstack
from numpy.linalg import norm
from numba import njit

@njit
def TNW_rotation(r, v):
    T = v / norm(v)  # T方向：速度矢量单位化
    h = cross(r, v)  # 轨道角动量
    N = h / norm(h)  # N方向：角动量单位化
    W = cross(N, T)  # W方向：完成右手系
    return vstack((T, W, N))

@njit
def PQW_rotation_rv(r, v, GM):
    r_norm = norm(r)
    h = cross(r, v)
    h_norm = norm(h)
    W = h / h_norm
    e = cross(v, h) / GM - r / r_norm
    e_norm = norm(e)
    P = e / e_norm
    Q = cross(W, P)
    return vstack((P, Q, W))

@njit
def PQW_rotation_coe(inc, raan, argp):
    R_raan = array([[cos(raan), sin(raan), 0.],
                    [-sin(raan), cos(raan), 0.],
                    [0., 0., 1.]], dtype=float64)
    R_inc = array([[1., 0., 0.],
                   [0., cos(inc), sin(inc)],
                   [0., -sin(inc), cos(inc)]], dtype=float64)
    R_argp = array([[cos(argp), sin(argp), 0.],
                    [-sin(argp), cos(argp), 0.],
                    [0., 0., 1.]], dtype=float64)
    return dot(dot(R_argp, R_inc), R_raan)