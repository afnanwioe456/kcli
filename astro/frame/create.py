from __future__ import annotations
import numpy as np
from astropy import units as u

from ..core.rotation import *
from ..orbit.create import Orbit
from ..body import *


class ReferenceFrame:
    def __init__(self,
                 parent: ReferenceFrame | None,
                 origin: np.ndarray = np.zeros(3) * u.km, 
                 rotation: np.ndarray = np.eye(3), 
                 velocity: np.ndarray = np.zeros(3) * u.km / u.s, 
                 angular_velocity: np.ndarray = np.zeros(3) * u.rad / u.s,
                 epoch: u.Quantity = 0 * u.s
                 ):
        """构建参考系

        Args:
            parent (ReferenceFrame): 父参考系
            origin (ndarray[Quantity], optional): 相对于父参考系的原点位置向量. Defaults to np.zeros(3).
            rotation (ndarray[Quantity], optional): 向父参考系转换的方向余弦矩阵. Defaults to np.eye(3).
            velocity (ndarray[Quantity], optional): 参考系平移速度向量. Defaults to np.zeros(3).
            angular_velocity (ndarray[Quantity], optional): 参考系的角速度. Defaults to np.zeros(3).
            epoch (Quantity): KSPRO历元时刻, 用于确定天体相对位置. Defaults to 0.
        """
        self.parent = parent
        self.origin = origin
        self.rotation = rotation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.epoch = epoch
        
    def transform_d_to_parent(self, direction):
        """将方向矢量转移到父参考系"""
        return self.rotation.T @ direction

    def transform_d_from_parent(self, direction):
        """将方向矢量从父参考系转移到此参考系"""
        return self.rotation @ direction
        
    def transform_p_to_parent(self, position):
        """将位置矢量转移到父参考系"""
        rotation = self.rotation.T
        return rotation @ position + self.origin

    def transform_p_from_parent(self, position):
        """将位置矢量从父参考系转移到此参考系"""
        rotation = self.rotation
        return rotation @ (position - self.origin)

    def transform_v_to_parent(self, rel_position, velocity):
        """将速度矢量转移到父参考系"""
        rotation = self.rotation.T
        coriolis_effect = np.cross(self.angular_velocity, rel_position)
        return rotation @ velocity + self.velocity + coriolis_effect

    def transform_v_from_parent(self, rel_position, velocity):
        """将速度矢量从父参考系转移到此参考系"""
        rotation = self.rotation
        coriolis_effect = np.cross(self.angular_velocity, rel_position)
        return rotation @ (velocity - self.velocity - coriolis_effect)

    @staticmethod
    def transform_d(direction, from_ref, to_ref):
        from_parents, to_parents = ReferenceFrame._transform_helper(from_ref, to_ref)
        for f in from_parents[:-1]:
            direction = f.transform_d_to_parent(direction)
        if len(to_parents) == 1:
            return direction
        for t in to_parents[-2::-1]:
            direction = t.transform_d_from_parent(direction)
        return direction

    @staticmethod
    def transform_p(position, from_ref, to_ref):
        from_parents, to_parents = ReferenceFrame._transform_helper(from_ref, to_ref)
        for f in from_parents[:-1]:
            position = f.transform_p_to_parent(position)
        if len(to_parents) == 1:
            return position
        for t in to_parents[-2::-1]:
            position = t.transform_p_from_parent(position)
        return position

    @staticmethod
    def transform_v(rel_position, velocity, from_ref, to_ref):
        from_parents, to_parents = ReferenceFrame._transform_helper(from_ref, to_ref)
        for f in from_parents[:-1]:
            velocity = f.transform_v_to_parent(rel_position, velocity)
            rel_position = f.transform_p_to_parent(rel_position)
        if len(to_parents) == 1:
            return velocity
        for t in to_parents[-2::-1]:
            velocity = t.transform_v_from_parent(rel_position, velocity)
            rel_position = f.transform_p_from_parent(rel_position)
        return velocity
        
    @staticmethod
    def _transform_helper(from_ref, to_ref):
        from_parents = ReferenceFrame._get_parent_refs(from_ref)
        to_parents = ReferenceFrame._get_parent_refs(to_ref)
        flag = False
        # LCA寻找最近公共祖先
        for i in range(min(len(from_parents), len(to_parents)) - 1, -1, -1):
            f = from_parents[i]
            t = to_parents[i]
            if isinstance(f, BCIFrame) and isinstance(t, BCIFrame) and f.body is t.body:
                flag = True
                break
        if flag:
            return from_parents[:i + 1], to_parents[:i + 1]
        raise ValueError("No common ancestor reference frame found!")

    @staticmethod
    def _get_parent_refs(ref: ReferenceFrame) -> list[ReferenceFrame]:
        parents = []
        while ref is not None:
            parents.append(ref)
            ref = ref.parent
        return parents
    
    @classmethod
    def transform_d_from_left_hand(cls, vector):
        return np.array([vector[0], vector[2], vector[1]], dtype=np.float64)

    @classmethod
    def transform_d_to_left_hand(cls, vector):
        return cls.transform_d_from_left_hand(vector)


SCI = ReferenceFrame(None)


class BCIFrame(ReferenceFrame):
    """天体中心惯性参考系"""
    def __init__(self, 
                 body: Body, 
                 epoch: u.Quantity):
        self.body = body
        self._parent = None
        self._origin = None
        self._velocity = None
        self.epoch = epoch

    def _sync(self):
        orb = self.body.orbit.propagate_to_epoch(self.epoch)
        self._origin = orb.r_vec
        self._velocity = orb.v_vec

    @property
    def parent(self):
        if self.body is KSP_Sun:
            return None
        if self._parent is None:
            self._parent = BCIFrame(self.body.attractor, self.epoch)
        return self._parent

    @property
    def origin(self):
        if self._origin is None:
            self._sync()
        return self._origin

    @property
    def velocity(self):
        if self._velocity is None:
            self._sync()
        return self._velocity

    @property
    def rotation(self):
        return np.eye(3)
    
    @property
    def angular_velocity(self):
        return np.zeros(3) / u.s


class ENUFrame(ReferenceFrame):
    @classmethod
    def from_bci_position(self, body: Body, position: u.Quantity, epoch: u.Quantity):
        rotation = ENU_rotation(position.to_value(u.km))
        body_angular_velocity = body.angular_velocity
        velocity = np.cross(body_angular_velocity, position)
        angular_velocity = rotation.T @ body_angular_velocity
        return ENUFrame(
            BCIFrame(body, epoch),
            position,
            rotation,
            velocity,
            angular_velocity,
            epoch
        )
        

class TNWFrame(ReferenceFrame):
    """切向-径向-法向参考系"""
    @staticmethod
    def from_orbit(orbit: Orbit):
        """
        从轨道建立切向-径向-法向参考系

        Args:
            orbit (Orbit): 轨道

        Returns:
            TNWFrame: 轨道参考系
        """
        parent = BCIFrame(orbit.attractor, orbit.epoch)
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        h = np.cross(position, velocity)
        rotation = TNW_rotation(position, velocity)
        angular_velocity = h / np.linalg.norm(position) ** 2
        return TNWFrame(
            parent,
            position * u.km, 
            rotation, 
            velocity * u.km / u.s, 
            angular_velocity / u.s,
            orbit.epoch
        )

    @classmethod
    def transform_d_from_left_hand(cls, vector):
        vector = np.array(vector, dtype=np.float64)
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)
        

class OrbitalFrame(ReferenceFrame):
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
        parent = BCIFrame(orbit.attractor, orbit.epoch)
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        rotation = TNW_rotation(position, velocity)
        return OrbitalFrame(
            parent,
            position * u.km, 
            rotation, 
            velocity * u.km / u.s,
            epoch = orbit.epoch
        )

    @classmethod
    def transform_d_from_left_hand(cls, vector):
        vector = np.array(vector, dtype=np.float64)
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class LocalAttitudeFrame(ReferenceFrame):
    """
    切向-径向-法向姿态参考系, 固定在轨道上
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        parent = BCIFrame(orbit.attractor, orbit.epoch)
        position = orbit.r_vec.to_value(u.km)
        velocity = orbit.v_vec.to_value(u.km / u.s)
        rotation = TNW_rotation(position, velocity)
        return LocalAttitudeFrame(
            parent,
            position * u.km, 
            rotation,
            epoch = orbit.epoch
        )

    @classmethod
    def transform_d_from_left_hand(cls, vector):
        vector = np.array(vector, dtype=np.float64)
        return np.array([vector[1], vector[0], vector[2]], dtype=np.float64)


class PQWFrame(ReferenceFrame):
    """
    近焦点参考系, 原点位于引力中心, p轴指向近地点, q轴指向真近点90°角, w轴指向角动量方向
    """
    @staticmethod
    def from_orbit(orbit: Orbit):
        parent = BCIFrame(orbit.attractor, orbit.epoch)
        r = orbit.r_vec.to_value(u.km)
        v = orbit.v_vec.to_value(u.km / u.s)
        GM = orbit.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
        rotation = PQW_rotation_rv(r, v, GM)
        return PQWFrame(parent, rotation=rotation)

    @staticmethod
    def from_vectors(p, q, w, parent: ReferenceFrame):
        rotation = np.vstack([p, q, w])
        return PQWFrame(parent, rotation=rotation, epoch=parent.epoch)



