import numpy as np
import krpc
from ..utils import UTIL_CONN


class PIDController:
    def __init__(self, Kp, Ki, Kd, windup_guard=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.windup_guard = windup_guard

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        self.integral = min(max(self.integral, -self.windup_guard), self.windup_guard)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class PIDVController:
    def __init__(self, Kp, Ki, Kd, windup_guard=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.windup_guard = windup_guard

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_guard, self.windup_guard)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


def smooth_step(x, a):
    """
    Tanh-like smooth step function.
    
    Args:
        x (float or ndarray): Input value(s).
        a (float): Threshold value where transition happens.
    
    Returns:
        ndarray: The output of the smooth step function.
    """
    return (np.tanh(x / a - 1) - np.tanh(x / a + 1)) / 2 + 1
