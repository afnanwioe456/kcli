from numba import njit

@njit
def clip(x, low, high):
    return min(max(x, low), high)

@njit
def lerp(a, b, alpha):
    return (b - a) * alpha + a