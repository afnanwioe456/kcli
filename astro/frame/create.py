from __future__ import annotations
import numpy as np
from astropy import units as u

from ..orbit.create import Orbit


class ReferenceFrameBase:
    def __init__(self,
                 father: ReferenceFrameBase,
                 origin=np.zeros(3), 
                 orientation=np.eye(3), 
                 velocity=np.zeros(3), 
                 angular_velocity=np.zeros(3),
                 ):
        """构建参考系

        Args:
            father (ReferenceFrame): 父参考系
            origin (NDArray[float64], optional): 相对于父参考系的原点位置向量. Defaults to np.zeros(3).
            orientation (NDArray[float64], optional): 向父参考系转换的方向余弦矩阵. Defaults to np.eye(3).
            velocity (NDArray[float64], optional): 参考系平移速度向量. Defaults to np.zeros(3).
            angular_velocity (NDArray[float64], optional): 参考系的角速度. Defaults to np.zeros(3).
        """
        self.father = father
        self.origin = np.array(origin)
        self.orientation = np.array(orientation)
        self.velocity = np.array(velocity)
        self.angular_velocity = np.array(angular_velocity)
        
    def transform_position_to_father_frame(self, position):
        """将位置矢量转移到父参考系

        Args:
            target_frame (ReferenceFrame): 目标参考系
            position (NDArray[float64]): 位置矢量

        Returns:
            NDArray: 目标参考系中的位置矢量
        """
        position = np.array(position)
        translation = self.origin - self.father.origin
        rotation = self.orientation
        return rotation @ position + translation

    def transform_position_from_father_frame(self, position):
        position = np.array(position)
        translation = self.origin - self.father.origin
        rotation = self.orientation.T
        return rotation @ (position - translation)

    def transform_velocity_to_father_frame(self, position_inref, velocity):
        position_inref = np.array(position_inref)
        velocity = np.array(velocity)
        relative_velocity = self.velocity - self.father.velocity
        rotation = self.orientation
        coriolis_effect = np.cross(self.angular_velocity, position_inref)
        return rotation @ (velocity + coriolis_effect) + relative_velocity

    def transform_velocity_from_father_frame(self, position_inref, velocity):
        position_inref = np.array(position_inref)
        velocity = np.array(velocity)
        relative_velocity = self.velocity - self.father.velocity
        rotation = self.orientation.T
        coriolis_effect = np.cross(self.angular_velocity, position_inref)
        return rotation @ (velocity - relative_velocity) - coriolis_effect
    
    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[0], vector[2], vector[1]], dtype=np.float64)


ECI = ReferenceFrameBase(None)


class TNWFrame(ReferenceFrameBase):
    """
    切向-法向-径向参考系
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        """
        从轨道建立切向-法向-径向参考系

        Args:
            orbit (Orbit): 轨道

        Returns:
            TNWFrame: 轨道参考系
        """
        position = orbit.r
        velocity = orbit.v

        T = velocity / np.linalg.norm(velocity)  # T方向：速度矢量单位化
        h = np.cross(position, velocity)  # 轨道角动量
        N = h / np.linalg.norm(h)  # N方向：角动量单位化
        W = np.cross(N, T)  # W方向：完成右手系

        orientation = np.vstack((T, W, N)).T
        angular_velocity = h / np.linalg.norm(position) ** 2
        return TNWFrame(ECI, position, orientation, velocity, angular_velocity)

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)
        

class OrbitalFrame(ReferenceFrameBase):
    """
    切向-法向-径向参考系, 但没有角速度, 即krpc orbital_reference_frame 和 ksp orbit_mode
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        """
        从轨道建立切向-法向-径向参考系, 但没有角速度,
        即krpc orbital_reference_frame 和 ksp orbit_mode

        Args:
            orbit (Orbit): 轨道

        Returns:
            OrbitalFrame: 轨道参考系
        """
        position = orbit.r
        velocity = orbit.v

        T = velocity / np.linalg.norm(velocity)
        h = np.cross(position, velocity)
        N = h / np.linalg.norm(h)
        W = np.cross(N, T)

        orientation = np.vstack((T, W, N)).T
        return TNWFrame(ECI, position, orientation, velocity)

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class LocalAttitudeFrame(ReferenceFrameBase):
    """
    切向-法向-径向姿态参考系, 固定在轨道上
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        position = orbit.r
        velocity = orbit.v

        T = velocity / np.linalg.norm(velocity)
        h = np.cross(position, velocity)
        N = h / np.linalg.norm(h)
        W = np.cross(N, T)

        orientation = np.vstack((T, W, N)).T
        return TNWFrame(ECI, position, orientation)

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class PQWFrame(ReferenceFrameBase):
    """
    PWQ参考系, 原点位于引力中心, p轴指向近地点, q轴指向真近点90°角, w轴指向角动量方向
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        r = orbit.r
        r_norm = np.linalg.norm(r)
        v = orbit.v
        
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        W = h / h_norm
        GM = orbit._poliorbit.attractor.k.to_value(u.m ** 3 / u.s ** 2)
        e = np.cross(v, h) / GM - r / r_norm
        e_norm = np.linalg.norm(e)
        P = e / e_norm
        Q = np.cross(W, P)

        orientation = np.vstack((P, Q, W)).T
        return PQWFrame(ECI, orientation=orientation)

