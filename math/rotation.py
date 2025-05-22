import numpy as np
import numpy.linalg as npl
from numba import njit

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
    axis = axis / npl.norm(axis)
    half_theta = theta / 2
    w = np.cos(half_theta)
    x, y, z = axis * np.sin(half_theta)
    return (x, y, z, w)

@njit
def quat_rotation(v, axis, theta):
    return quat_rotations(v, [(axis, theta)])

@njit
def quat_rotations(v, rotations):
    total_q = (0.0, 0.0, 0.0, 1.0)
    for axis, theta in rotations:
        q = _to_quaternion(axis, theta)
        total_q = _quaternion_mul(total_q, q)

    v_quat = (v[0], v[1], v[2], 0.0)
    q_conj = _quaternion_conj(total_q)

    temp = _quaternion_mul(total_q, v_quat)
    v_rot_quat = _quaternion_mul(temp, q_conj)

    return np.array([v_rot_quat[0], v_rot_quat[1], v_rot_quat[2]], dtype=np.float64)

@njit
def vec_rotation(vec, n, theta):
    """
    使用 Rodrigues 旋转公式绕轴 n 旋转矢量 u0, 旋转角度 theta.
    """
    n = n / npl.norm(n)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return vec*cos_theta + np.cross(n, vec)*sin_theta + n*np.dot(n, vec)*(1-cos_theta)

@njit
def solve_rotation_angle(v, u, n, angle):
    """
    计算使向量v绕单位轴n旋转一定角度后, 与向量u成angle的夹角所需的旋转角度phi(0~2pi)
    """
    v = v / npl.norm(v)
    u = u / npl.norm(u)
    n = n / npl.norm(n)

    A = np.dot(u, v)
    B = np.dot(u, np.cross(n, v))
    C = np.dot(n, v) * np.dot(u, n)
    D = A - C
    E = np.cos(angle)

    R = np.hypot(D, B)
    rhs = E - C

    if np.abs(rhs) > R + 1e-10:
        return

    alpha = np.arctan2(B, D)
    delta = np.arccos(min(max(rhs / R, -1.0), 1.0))

    phi1 = alpha + delta
    phi2 = alpha - delta

    phi1 = phi1 % (2 * np.pi)
    phi2 = phi2 % (2 * np.pi)

    return phi1, phi2