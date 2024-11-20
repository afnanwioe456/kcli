from __future__ import annotations
import numpy as np

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

    def transform_velocity_to_father_frame(self, position, velocity):
        position = np.array(position)
        velocity = np.array(velocity)
        relative_velocity = self.velocity - self.father.velocity
        rotation = self.orientation
        coriolis_effect = np.cross(self.angular_velocity, position)
        return rotation @ velocity + relative_velocity + coriolis_effect

    def transform_velocity_from_father_frame(self, position, velocity):
        position = np.array(position)
        velocity = np.array(velocity)
        relative_velocity = self.velocity - self.father.velocity
        rotation = self.orientation.T
        coriolis_effect = np.cross(self.angular_velocity, position)
        return rotation @ (velocity - relative_velocity - coriolis_effect)
    

ECI = ReferenceFrameBase(None)


class TNWFrame(ReferenceFrameBase):
    @staticmethod
    def from_orbit(orbit: Orbit):
        """从轨道建立切向-法向-径向参考系, 即krpc中orbital_reference_frame的右手系

        Args:
            orbit (Orbit): 轨道

        Returns:
            OrbitalFrame: 轨道参考系
        """
        position = orbit.poliorbit.r * 1000
        velocity = orbit.poliorbit.v * 1000
        # TODO: 单位问题, 考虑自定义接口私有化poliorbit
        # TODO: 精度问题, 见测试, 考虑调用私有方法
        # TODO: 左手系问题, 避免了么？

        T = velocity / np.linalg.norm(velocity)  # T方向：速度矢量单位化
        h = np.cross(position, velocity)  # 轨道角动量
        N = h / np.linalg.norm(h)  # N方向：角动量单位化
        W = np.cross(N, T)  # W方向：完成右手系

        orientation = np.vstack((T, W, N)).T
        angular_velocity = h / np.linalg.norm(position) ** 2
        return TNWFrame(ECI, position, orientation, velocity, angular_velocity)

        
if __name__ == '__main__':
    from poliastro.twobody.orbit import Orbit as Poliorbit
    from poliastro.bodies import Earth
    from poliastro.plotting import OrbitPlotter3D
    from astropy import units as u
    import krpc
    
    conn = krpc.connect()
    vessel = conn.space_center.active_vessel
    target = conn.space_center.target_vessel
    
    v = vessel.velocity(vessel.orbit.body.non_rotating_reference_frame)
    v = np.array([v[0], v[2], v[1]], dtype=np.float64)
    r = vessel.position(vessel.orbit.body.non_rotating_reference_frame)
    r = np.array([r[0], r[2], r[1]], dtype=np.float64)

    tv_eci = target.velocity(vessel.orbit.body.non_rotating_reference_frame)
    tv_eci = np.array([tv_eci[0], tv_eci[2], tv_eci[1]], dtype=np.float64)
    tr_eci = target.position(vessel.orbit.body.non_rotating_reference_frame)
    tr_eci = np.array([tr_eci[0], tr_eci[2], tr_eci[1]], dtype=np.float64)
    tv_tnw = target.velocity(vessel.orbital_reference_frame)
    tv_tnw = np.array([tv_tnw[1], tv_tnw[0], tv_tnw[2]], dtype=np.float64)
    tr_tnw = target.position(vessel.orbital_reference_frame)
    tr_tnw = np.array([tr_tnw[1], tr_tnw[0], tr_tnw[2]], dtype=np.float64)
    print(tv_tnw, tr_tnw)

    orbit = Orbit.from_krpcv(vessel)
    tnw = TNWFrame.from_orbit(orbit)
    print(type(tnw.velocity), tnw.velocity, v)
    print(type(tnw.origin), tnw.origin, r)
    resv_eci = tnw.transform_velocity_to_father_frame(tr_tnw, tv_tnw)
    print(resv_eci)
    print(tv_eci)
