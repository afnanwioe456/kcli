import numpy as np
import numpy.linalg as npl
from numba import njit

from .stumpff import *
from .rotation import *
from ...math import vector as mv

@njit
def nu2r(nu, h, e, GM):
    denom = 1 + e * np.cos(nu)
    if abs(denom) < 1e-10:
        return np.inf
    return h ** 2 / (GM * denom)

@njit
def r2nu(r, h, e, GM, sign=True):
    if e < 1e-10:
        return 0
    nu = np.arccos((h ** 2 / GM / r - 1) / e)
    if not sign:
        nu = 2 * np.pi - nu
    return nu

@njit
def r2v(r, a, GM):
    return np.sqrt(- GM / a + 2 * GM / r)

@njit
def v2r(v, a, GM):
    return 2 * GM / (v ** 2 + GM / a)

@njit
def a2T(a, GM):
    """椭圆轨道的周期"""
    return 2 * np.pi * a ** 1.5 / GM ** 0.5

@njit
def T2v(T, r, GM):
    """根据椭圆轨道周期和半径求速度"""
    a = (T * np.sqrt(GM) / (2 * np.pi)) ** (2 / 3)
    return r2v(r, a, GM)

@njit
def nu2E(nu, e):
    """椭圆轨道真近点角求偏近点角"""
    return 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan(nu / 2))

@njit
def E2Me(E, e):
    """椭圆轨道偏近点角求平近点角"""
    return E - e * np.sin(E)
    
@njit
def nu2dt_e(nu, e, T):
    """椭圆轨道从近地点到给定真近点角处的时间"""
    E = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan(nu / 2))
    Me = E - e * np.sin(E)
    dt = Me / (np.pi * 2) * T
    return dt

@njit
def nu2F(nu, e):
    """双曲线轨道真近点角求偏近点角"""
    return 2 * np.arctanh(((e - 1) / (e + 1)) ** 0.5 * np.tan(nu / 2))

@njit
def F2Mh(F, e):
    """双曲线轨道偏近点角求平近点角"""
    return e * np.sinh(F) - F

@njit
def nu2dt_h(nu, e, GM, h):
    """双曲线轨道从近地点到给定真近点角处的时间"""
    F = 2 * np.arctanh(((e - 1) / (e + 1)) ** 0.5 * np.tan(nu / 2))
    Mh = e * np.sinh(F) - F
    dt = Mh * ((h ** 3 / GM ** 2) / (e ** 2 - 1) ** 1.5)
    return dt

@njit
def dt2Me(dt, T):
    """椭圆轨道由dt计算平近点角"""
    return (dt % T) * 2 * np.pi / T

@njit
def Me2nu(Me, e, tol=1e-8, max_iter=100):
    """椭圆轨道由平近点角求真近点角"""
    if Me < np.pi:
        E = Me + e / 2
    else:
        E = Me - e / 2
    ratio = 1
    step = 0
    while abs(ratio) > tol and step < max_iter:
        ratio = (E - e * np.sin(E) - Me) / (1 - e * np.cos(E))
        E -= ratio
        step += 1
    nu = 2 * np.arctan(np.tan(E / 2) / ((1 - e) / (1 + e)) ** 0.5)
    if nu < 0:
        nu += 2 * np.pi
    return nu

@njit
def dt2Mh(dt, e, GM, h):
    """双曲线轨道由dt计算平近点角"""
    return (GM ** 2 / h ** 3) * (e ** 2 - 1) ** 1.5 * dt

@njit
def Mh2nu(Mh, e, tol=1e-8, max_iter=100):
    """双曲线轨道由平近点角求真近点角"""
    F = Mh
    ratio = 1
    step = 0
    while abs(ratio) > tol and step < max_iter:
        ratio = (e * np.sinh(F) - F - Mh) / (e * np.cosh(F) - 1)
        F -= ratio
        step += 1
    nu = 2 * np.arctan(np.tanh(F / 2) / ((e - 1) / (e + 1)) ** 0.5)
    if nu < 0:
        nu += 2 * np.pi
    return nu

@njit
def rv2alpha(r, v, GM):
    """能量方程由rv标量求解alpha"""
    return 2 / r - v ** 2 / GM

@njit
def rv2uni_nu(r, vr, dt, alpha, GM, tol=1e-8, max_iter=100):
    """求全局真近点角

    Args:
        r (float): 半径, km
        vr (float): 速度径向分量, km/s
        dt (float): dt, s
        alpha (float): 全局长半轴的倒数(椭圆>0, 双曲线<0), /km
    """
    sqrt_GM = np.sqrt(GM)
    x = sqrt_GM * abs(alpha) * dt
    step = 0
    ratio = 1
    C1 = r * vr / sqrt_GM
    C2 = 1 - alpha * r
    C3 = sqrt_GM * dt
    while abs(ratio) > tol and step < max_iter:
        z = alpha * x ** 2
        C = stumpff_c(z)
        S = stumpff_s(z)
        F = C1 * x ** 2 * C + C2 * x ** 3 * S + r * x - C3
        dF = C1 * x * (1 - z * S) + C2 * x ** 2 * C + r
        ratio = F / dF
        x -= ratio
        step += 1
    return x

@njit
def h2a(h, GM, e):
    return h ** 2 / GM / (1 - e ** 2)

@njit
def a2h(a, GM, e):
    s = a * GM * (1 - e ** 2)
    return np.sqrt(s)

@njit
def rv2coe(r_vec, v_vec, GM):
    """由状态向量计算经典轨道根数(h, e, inc, raan, argp, nu)"""
    eps = 1e-10
    r = npl.norm(r_vec)
    v = npl.norm(v_vec)

    # 比角动量
    h_vec = np.cross(r_vec, v_vec)
    h = npl.norm(h_vec)

    # 倾角
    inc = np.arccos(h_vec[2] / h)

    # 升交点向量
    K = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    S = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    N_vec = np.cross(K, h_vec)
    N = npl.norm(N_vec)

    # 升交点赤经（RAAN）
    if N > eps:
        raan = mv.angle_between_vectors(S, N_vec, K)
    else:
        # 赤道轨道
        raan = 0.0

    # 偏心率向量
    e_vec = ((v ** 2 - GM / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / GM
    e = npl.norm(e_vec)

    # 近地点幅角（argp）与真近点角（nu）
    if e > eps:
        # 椭圆/双曲线
        argp = mv.angle_between_vectors(N_vec, e_vec, h_vec)
        nu = mv.angle_between_vectors(e_vec, r_vec, h_vec)
    else:
        # 圆轨道
        argp = 0.0
        if N > eps:
            # 有倾角的圆轨道：使用纬度角
            nu = mv.angle_between_vectors(N_vec, r_vec, h_vec)
        else:
            # 圆且赤道轨道：使用真近点经度
            nu = mv.angle_between_vectors(S, r_vec, h_vec)

    return h, e, inc, raan, argp, nu

@njit
def rv2he(r_vec, v_vec, GM):
    r = npl.norm(r_vec)
    v = npl.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    e_vec = ((v ** 2 - GM / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / GM
    return npl.norm(h_vec), npl.norm(e_vec)
        
@njit
def coe2rv(h, e, inc, raan, argp, nu, GM):
    pi = np.array([1., 0., 0.], dtype=np.float64)
    qi = np.array([0., 1., 0.], dtype=np.float64)
    r_vec = (h ** 2 / GM) * (1 / (1 + e * np.cos(nu))) * (np.cos(nu) * pi + np.sin(nu) * qi)
    v_vec = (GM / h) * (-np.sin(nu) * pi + ((e + np.cos(nu)) * qi))
    rotation = PQW_rotation_coe(inc, raan, argp).T
    r_vec = np.dot(rotation, r_vec)
    v_vec = np.dot(rotation, v_vec)
    return r_vec, v_vec