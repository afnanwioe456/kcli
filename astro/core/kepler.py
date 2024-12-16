from numpy import cos, arccos, sin, pi, sqrt, tan, arctan, arctanh, sinh, cosh, tanh, dot, cross, float64, array, inf
from numpy.linalg import norm
from numba import njit

from .stumpff import *
from .rotation import *

@njit
def nu2r(nu, h, e, GM):
    if cos(nu) <= -1 / e:
        return inf
    return h ** 2 / GM / (1 + e * cos(nu))

@njit
def r2nu(r, h, e, GM, sign=True):
    nu = arccos((h ** 2 / GM / r - 1) / e)
    if not sign:
        nu = 2 * pi - nu
    return nu

@njit
def r2v(r, a, GM, e):
    if e > 1:
        a = -a
    return sqrt(- GM / a + 2 * GM / r)

@njit
def v2r(v, a, GM, e):
    if e > 1:
        a = -a
    return 2 * GM / (v ** 2 + GM / a)

@njit
def period(a, GM):
    """椭圆轨道的周期"""
    return 2 * pi * a ** 1.5 / GM ** 0.5

@njit
def nu2E(nu, e):
    """椭圆轨道真近点角求偏近点角"""
    return 2 * arctan(((1 - e) / (1 + e)) ** 0.5 * tan(nu / 2))

@njit
def E2Me(E, e):
    """椭圆轨道偏近点角求平近点角"""
    return E - e * sin(E)
    
@njit
def nu2dt_e(nu, e, T):
    """椭圆轨道从近地点到给定真近点角处的时间"""
    E = 2 * arctan(((1 - e) / (1 + e)) ** 0.5 * tan(nu / 2))
    Me = E - e * sin(E)
    dt = Me / (pi * 2) * T
    return dt

@njit
def nu2F(nu, e):
    """双曲线轨道真近点角求偏近点角"""
    return 2 * arctanh(((e - 1) / (e + 1)) ** 0.5 * tan(nu / 2))

@njit
def F2Mh(F, e):
    """双曲线轨道偏近点角求平近点角"""
    return e * sinh(F) - F

@njit
def nu2dt_h(nu, e, GM, h):
    """双曲线轨道从近地点到给定真近点角处的时间"""
    F = 2 * arctanh(((e - 1) / (e + 1)) ** 0.5 * tan(nu / 2))
    Mh = e * sinh(F) - F
    dt = Mh * ((h ** 3 / GM ** 2) / (e ** 2 - 1) ** 1.5)
    return dt

@njit
def dt2Me(dt, T):
    """椭圆轨道由dt计算平近点角"""
    return (dt % T) * 2 * pi / T

@njit
def Me2nu(Me, e, tol=1e-8, max_iter=100):
    """椭圆轨道由平近点角求真近点角"""
    if Me < pi:
        E = Me + e / 2
    else:
        E = Me - e / 2
    ratio = 1
    step = 0
    while abs(ratio) > tol and step < max_iter:
        ratio = (E - e * sin(E) - Me) / (1 - e * cos(E))
        E -= ratio
        step += 1
    nu = 2 * arctan(tan(E / 2) / ((1 - e) / (1 + e)) ** 0.5)
    if nu < 0:
        nu += 2 * pi
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
        ratio = (e * sinh(F) - F - Mh) / (e * cosh(F) - 1)
        F -= ratio
        step += 1
    nu = 2 * arctan(tanh(F / 2) / ((e - 1) / (e + 1)) ** 0.5)
    if nu < 0:
        nu += 2 * pi
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
    sqrt_GM = sqrt(GM)
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
    return h ** 2 / GM / abs(1 - e ** 2)

@njit
def a2h(a, GM, e):
    return sqrt(a * GM * abs(e ** 2 - 1))

@njit
def rv2coe(r_vec, v_vec, GM):
    """由状态向量计算经典轨道根数(h, e, inc, raan, argp, nu)"""
    r = norm(r_vec)
    v = norm(v_vec)
    vr = dot(r_vec, v_vec) / r
    h_vec = cross(r_vec, v_vec)
    h = norm(h_vec)
    inc = arccos(h_vec[2] / h)
    K = array([0, 0, 1], dtype=float64)
    N_vec = cross(K, h_vec)
    N = norm(N_vec)
    raan = 0 if N == 0 else arccos(N_vec[0] / N)
    if N_vec[1] < 0:
        raan = 2 * pi - raan
    e_vec = ((v ** 2 - GM / r) * r_vec - dot(r_vec, v_vec) * v_vec) / GM
    e = norm(e_vec)
    argp = 0 if N == 0 else arccos(dot(N_vec, e_vec) / (N * e))
    if e_vec[2] < 0:
        argp = 2 * pi - argp
    nu = arccos(max(min((h ** 2 / (GM * r) - 1) / e, 1), -1))
    if vr < 0:
        nu = 2 * pi - nu
    return h, e, inc, raan, argp, nu
        
@njit
def coe2rv(h, e, inc, raan, argp, nu, GM):
    pi = array([1., 0., 0.], dtype=float64)
    qi = array([0., 1., 0.], dtype=float64)
    r_vec = (h ** 2 / GM) * (1 / (1 + e * cos(nu))) * (cos(nu) * pi + sin(nu) * qi)
    v_vec = (GM / h) * (-sin(nu) * pi + ((e + cos(nu)) * qi))
    rotation = PQW_rotation_coe(inc, raan, argp).T
    r_vec = dot(rotation, r_vec)
    v_vec = dot(rotation, v_vec)
    return r_vec, v_vec