from __future__ import annotations
import numpy as np
from astropy import units as u

from ..core.rotation import *
from ..orbit.create import Orbit


class ReferenceFrameBase:
    def __init__(self,
                 parent: ReferenceFrameBase,
                 origin=np.zeros(3) * u.km, 
                 rotation=np.eye(3), 
                 velocity=np.zeros(3) * u.km / u.s, 
                 angular_velocity=np.zeros(3) / u.s,
                 ):
        """构建参考系

        Args:
            parent (ReferenceFrame): 父参考系
            origin (ndarray[Quantity], optional): 相对于父参考系的原点位置向量. Defaults to np.zeros(3).
            rotation (ndarray[Quantity], optional): 向父参考系转换的方向余弦矩阵. Defaults to np.eye(3).
            velocity (ndarray[Quantity], optional): 参考系平移速度向量. Defaults to np.zeros(3).
            angular_velocity (ndarray[Quantity], optional): 参考系的角速度. Defaults to np.zeros(3).
        """
        self.parent = parent
        self.origin = origin
        self.rotation = rotation
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def transform_d_to_parent(self, direction):
        """将方向矢量转移到父参考系"""
        return self.rotation.T @ direction

    def transform_d_from_parent(self, direction):
        """将方向矢量从父参考系转移到此参考系"""
        return self.rotation @ direction
        
    @u.quantity_input(position=u.km)
    def transform_p_to_parent(self, position):
        """将位置矢量转移到父参考系"""
        translation = self.origin - self.parent.origin
        rotation = self.rotation.T
        return rotation @ position + translation

    @u.quantity_input(position=u.km)
    def transform_p_from_parent(self, position):
        """将位置矢量从父参考系转移到此参考系"""
        translation = self.origin - self.parent.origin
        rotation = self.rotation
        return rotation @ (position - translation)

    @u.quantity_input(position_inref=u.km, velocity=u.km/u.s)
    def transform_v_to_parent(self, rel_position, velocity):
        """将速度矢量转移到父参考系"""
        relative_velocity = self.velocity - self.parent.velocity
        rotation = self.rotation.T
        coriolis_effect = np.cross(self.angular_velocity, rel_position)
        return rotation @ velocity + coriolis_effect + relative_velocity

    @u.quantity_input(position_inref=u.km, velocity=u.km/u.s)
    def transform_v_from_parent(self, rel_position, velocity):
        """将速度矢量从父参考系转移到此参考系"""
        relative_velocity = self.velocity - self.parent.velocity
        rotation = self.rotation
        coriolis_effect = np.cross(self.angular_velocity, rel_position)
        return rotation @ (velocity - relative_velocity - coriolis_effect)
    
    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[0], vector[2], vector[1]], dtype=np.float64)


ECI = ReferenceFrameBase(None)


class TNWFrame(ReferenceFrameBase):
    """
    切向-径向-法向参考系
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        """
        从轨道建立切向-径向-法向参考系

        Args:
            orbit (Orbit): 轨道

        Returns:
            TNWFrame: 轨道参考系
        """
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        h = np.cross(position, velocity)
        rotation = TNW_rotation(position, velocity)
        angular_velocity = h / np.linalg.norm(position) ** 2
        return TNWFrame(
            ECI, 
            position * u.km, 
            rotation, 
            velocity * u.km / u.s, 
            angular_velocity / u.s
        )

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)
        

class OrbitalFrame(ReferenceFrameBase):
    """
    切向-径向-法向参考系, 但没有角速度, 即krpc orbital_reference_frame 和 ksp orbit_mode
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        """
        从轨道建立切向-径向-法向参考系, 但没有角速度,
        即krpc orbital_reference_frame 和 ksp orbit_mode

        Args:
            orbit (Orbit): 轨道

        Returns:
            OrbitalFrame: 轨道参考系
        """
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        rotation = TNW_rotation(position, velocity)
        return OrbitalFrame(
            ECI, 
            position * u.km, 
            rotation, 
            velocity * u.km / u.s
        )

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class LocalAttitudeFrame(ReferenceFrameBase):
    """
    切向-径向-法向姿态参考系, 固定在轨道上
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        rotation = TNW_rotation(position, velocity)
        return LocalAttitudeFrame(
            ECI, 
            position * u.km, 
            rotation
        )

    @staticmethod
    def from_left_hand(vector):
        vector = np.array(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != 3:
            raise ValueError(f"Shape of the vector should be (,3), got {vector.shape}")
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class PQWFrame(ReferenceFrameBase):
    """
    近焦点参考系, 原点位于引力中心, p轴指向近地点, q轴指向真近点90°角, w轴指向角动量方向
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        r = orbit.r_vec.to_value(u.km)
        v = orbit.v_vec.to_value(u.km / u.s)
        GM = orbit.attractor.k.to_value(u.km ** 3 / u.s ** 2)
        rotation = PQW_rotation_rv(r, v, GM)
        return PQWFrame(ECI, rotation=rotation)

    @staticmethod
    def from_vectors(p, q, w):
        rotation = np.vstack([p, q, w])
        return PQWFrame(ECI, rotation=rotation)

