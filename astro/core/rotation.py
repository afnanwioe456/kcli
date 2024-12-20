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

@njit
def _quaternion_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)

@njit
def _quaternion_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

@njit
def _to_quaternion(axis, theta):
    axis = axis / norm(axis)
    half_theta = theta / 2
    w = cos(half_theta)
    x, y, z = axis * sin(half_theta)
    return (x, y, z, w)

@njit
def axis_rotation(v, axis, theta):
    return axis_rotations(v, [(axis, theta)])

@njit
def axis_rotations(v, rotations):
    total_quaternion = (0.0, 0.0, 0.0, 1.0)
    for axis, theta in rotations:
        q = _to_quaternion(axis, theta)
        total_quaternion = _quaternion_mul(total_quaternion, q)
    v_quat = (v[0], v[1], v[2], 0.0)
    q_conj = _quaternion_conj(q)
    temp = _quaternion_mul(q, v_quat)
    v_rot_quat = _quaternion_mul(temp, q_conj)
    return array([v_rot_quat[0], v_rot_quat[1], v_rot_quat[2]], dtype=float64)