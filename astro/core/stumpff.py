from numpy import cos, sinh, sin, sqrt, cosh
from numba import njit


@njit
def stumpff_s(z):
    if z > 0:
        return (sqrt(z) - sin(sqrt(z))) / (sqrt(z)) ** 3
    elif z < 0:
        return (sinh(sqrt(-z)) - sqrt(-z)) / (sqrt(-z)) ** 3
    else:
        return 1 / 6
    
@njit
def stumpff_c(z):
    if z > 0:
        return (1 - cos(sqrt(z))) / z
    elif z < 0:
        return (cosh(sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2