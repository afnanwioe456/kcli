import numpy as np
import numpy.linalg as npl
from numba import njit

@njit
def izzo(k, r1, r2, tof, M=0, prograde=True, lowpath=True, numiter=35, rtol=1e-8):
    """Applies izzo algorithm to solve Lambert's problem.

    Parameters
    ----------
    k : float
        Gravitational Constant
    r1 : numpy.ndarray
        Initial position vector
    r2 : numpy.ndarray
        Final position vector
    tof : float
        Time of flight between both positions
    M : int
        Number of revolutions
    prograde: boolean
        Controls the desired inclination of the transfer orbit.
    lowpath: boolean
        If `True` or `False`, gets the transfer orbit whose vacant focus is
        below or above the chord line, respectively.
    numiter : int
        Number of iterations
    rtol : float
        Error tolerance

    Returns
    -------
    v1: numpy.ndarray
        Initial velocity vector
    v2: numpy.ndarray
        Final velocity vector

    """

    # Check preconditions
    assert tof > 0
    assert k > 0

    # Check collinearity of r1 and r2
    if not np.cross(r1, r2).any():
        raise ValueError(
            "Lambert solution cannot be computed for collinear vectors"
        )

    # Chord
    c = r2 - r1
    c_norm, r1_norm, r2_norm = np.norm(c), np.norm(r1), np.norm(r2)

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Versors
    i_r1, i_r2 = r1 / r1_norm, r2 / r2_norm
    i_h = np.cross(i_r1, i_r2)
    i_h = i_h / np.norm(i_h)  # Fixed from paper

    # Geometry of the problem
    ll = np.sqrt(1 - min(1.0, c_norm / s))

    # Compute the fundamental tangential directions
    if i_h[2] < 0:
        ll = -ll
        i_t1, i_t2 = np.cross(i_r1, i_h), np.cross(i_r2, i_h)
    else:
        i_t1, i_t2 = np.cross(i_h, i_r1), np.cross(i_h, i_r2)

    # Correct transfer angle parameter and tangential vectors if required
    ll, i_t1, i_t2 = (ll, i_t1, i_t2) if prograde else (-ll, -i_t1, -i_t2)

    # Non dimensional time of flight
    T = np.sqrt(2 * k / s**3) * tof

    # Find solutions
    x, y = _find_xy(ll, T, M, numiter, lowpath, rtol)

    # Reconstruct
    gamma = np.sqrt(k * s / 2)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = np.sqrt(1 - rho**2)

    # Compute the radial and tangential components at r0 and r
    V_r1, V_r2, V_t1, V_t2 = _reconstruct(
        x, y, r1_norm, r2_norm, ll, gamma, rho, sigma
    )

    # Solve for the initial and final velocity
    v1 = V_r1 * (r1 / r1_norm) + V_t1 * i_t1
    v2 = V_r2 * (r2 / r2_norm) + V_t2 * i_t2

    return v1, v2


@njit
def _reconstruct(x, y, r1, r2, ll, gamma, rho, sigma):
    """Reconstruct solution velocity vectors."""
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2
    V_t1 = gamma * sigma * (y + ll * x) / r1
    V_t2 = gamma * sigma * (y + ll * x) / r2
    return V_r1, V_r2, V_t1, V_t2


@njit
def _find_xy(ll, T, M, numiter, lowpath, rtol):
    """Computes all x, y for given number of revolutions."""
    # For abs(ll) == 1 the derivative is not continuous
    assert abs(ll) < 1
    assert T > 0  # Mistake in the original paper

    M_max = np.floor(T / np.pi)
    T_00 = np.arccos(ll) + ll * np.sqrt(1 - ll**2)  # T_xM

    # Refine maximum number of revolutions if necessary
    if T < T_00 + M_max * np.pi and M_max > 0:
        _, T_min = _compute_T_min(ll, M_max, numiter, rtol)
        if T < T_min:
            M_max -= 1

    # Check if a feasible solution exist for the given number of revolutions
    # This departs from the original paper in that we do not compute all solutions
    if M > M_max:
        raise ValueError("No feasible solution, try lower M")

    # Initial guess
    x_0 = _initial_guess(T, ll, M, lowpath)

    # Start Householder iterations from x_0 and find x, y
    x = _householder(x_0, T, ll, M, rtol, numiter)
    y = _compute_y(x, ll)

    return x, y


@njit
def _compute_y(x, ll):
    """Computes y."""
    return np.sqrt(1 - ll**2 * (1 - x**2))


@njit
def _compute_psi(x, y, ll):
    """Computes psi.

    "The auxiliary angle psi is computed using Eq.(17) by the appropriate
    inverse function"

    """
    if -1 <= x < 1:
        # Elliptic motion
        # Use arc cosine to avoid numerical errors
        return np.arccos(x * y + ll * (1 - x**2))
    elif x > 1:
        # Hyperbolic motion
        # The hyperbolic sine is bijective
        return np.arcsinh((y - x * ll) * np.sqrt(x**2 - 1))
    else:
        # Parabolic motion
        return 0.0


@njit
def _tof_equation(x, T0, ll, M):
    """Time of flight equation."""
    return _tof_equation_y(x, _compute_y(x, ll), T0, ll, M)


@njit
def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computated y."""
    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * _hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4 * ll * eta) * 0.5
    else:
        psi = _compute_psi(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M * np.pi, np.sqrt(abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )

    return T_ - T0


@njit
def _tof_equation_p(x, y, T, ll):
    # TODO: What about derivatives when x approaches 1?
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)


@njit
def _tof_equation_p2(x, y, T, dT, ll):
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (
        1 - x**2
    )


@njit
def _tof_equation_p3(x, y, _, dT, ddT, ll):
    return (
        7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5
    ) / (1 - x**2)


@njit
def _compute_T_min(ll, M, numiter, rtol):
    """Compute minimum T."""
    if ll == 1:
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
    else:
        if M == 0:
            x_T_min = np.inf
            T_min = 0.0
        else:
            # Set x_i > 0 to avoid problems at ll = -1
            x_i = 0.1
            T_i = _tof_equation(x_i, 0.0, ll, M)
            x_T_min = _halley(x_i, T_i, ll, rtol, numiter)
            T_min = _tof_equation(x_T_min, 0.0, ll, M)

    return x_T_min, T_min


@njit
def _initial_guess(T, ll, M, lowpath):
    """Initial guess."""
    if M == 0:
        # Single revolution
        T_0 = np.arccos(ll) + ll * np.sqrt(1 - ll**2) + M * np.pi  # Equation 19
        T_1 = 2 * (1 - ll**3) / 3  # Equation 21
        if T >= T_0:
            x_0 = (T_0 / T) ** (2 / 3) - 1
        elif T < T_1:
            x_0 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll**5) + 1
        else:
            # This is the real condition, which is not exactly equivalent
            # elif T_1 < T < T_0
            # Corrected initial guess,
            # piecewise equation right after expression (30) in the original paper is incorrect
            # See https://github.com/poliastro/poliastro/issues/1362
            x_0 = np.exp(np.log(2) * np.log(T / T_0) / np.log(T_1 / T_0)) - 1

        return x_0
    else:
        # Multiple revolution
        x_0l = (((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) - 1) / (
            ((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) + 1
        )
        x_0r = (((8 * T) / (M * np.pi)) ** (2 / 3) - 1) / (
            ((8 * T) / (M * np.pi)) ** (2 / 3) + 1
        )

        # Select one of the solutions according to desired type of path
        x_0 = (
            max(np.array([x_0l, x_0r]))
            if lowpath
            else min(np.array([x_0l, x_0r]))
        )

        return x_0


@njit
def _halley(p0, T0, ll, tol, maxiter):
    """Find a minimum of time of flight equation using the Halley method.

    Notes
    -----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.

    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fder = _tof_equation_p(p0, y, T0, ll)
        fder2 = _tof_equation_p2(p0, y, T0, fder, ll)
        if fder2 == 0:
            raise RuntimeError("Derivative was zero")
        fder3 = _tof_equation_p3(p0, y, T0, fder, fder2, ll)

        # Halley step (cubic)
        p = p0 - 2 * fder * fder2 / (2 * fder2**2 - fder * fder3)

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")


@njit
def _householder(p0, T0, ll, M, tol, maxiter):
    """Find a zero of time of flight equation using the Householder method.

    Notes
    -----
    This function is private because it assumes a calling convention specific to
    this module and is not really reusable.

    """
    for ii in range(maxiter):
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        p = p0 - fval * (
            (fder**2 - fval * fder2 / 2)
            / (fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6)
        )

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError("Failed to converge")

@njit
def _hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].

    .. todo::
        Add more information about this function

    Notes
    -----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    """
    if x >= 1.0:
        return np.inf
    else:
        res = 1.0
        term = 1.0
        ii = 0
        while True:
            term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
            res_old = res
            res += term
            if res_old == res:
                return res
            ii += 1