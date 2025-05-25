import numpy as np

from .constants import KSP_BODY_CONSTANTS
from .utils import UTIL_CONN


class Body:
    _instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name, attractor_name, mass, mu, r, initial_rotation, 
                 rotational_period, soi, atmosphere_height, angular_velocity):
        # NOTE: 所有KSP天体轴倾一致, 因此我们只需要角速度矢量即可定义天体, 对于实际情况是不成立的
        if '_initialized' in self.__dict__:
            return
        self._initialized = True

        self.name               = name
        self._attractor_name    = attractor_name
        self.mass               = mass
        self.mu                 = mu
        self.r                  = r
        self.initial_rotation   = initial_rotation
        self.rotational_period  = rotational_period
        self.soi                = soi
        self.atmosphere_height  = atmosphere_height
        self.has_atmosphere     = atmosphere_height > 0
        self.angular_velocity   = np.asarray(angular_velocity, dtype=np.float64)

    @classmethod
    def get_or_create(cls, name):
        if name in cls._instances:
            return cls._instances[name]
        bc                      = KSP_BODY_CONSTANTS[name]
        attractor_name          = bc['attractor']
        mass                    = bc['mass']
        mu                      = bc['gravational_parameter']
        r                       = bc['equatorial_radius']
        initial_rotation        = bc['initial_rotation']
        rotational_period       = bc['rotational_period']
        soi                     = bc['sphere_of_influence']
        atmosphere_height       = bc['atmosphere_height']
        angular_velocity        = np.array(bc['angular_velocity'], dtype=np.float64)
        return cls(name, attractor_name, mass, mu, r, initial_rotation, 
                   rotational_period, soi, atmosphere_height, angular_velocity)

    @property
    def attractor(self):
        return Body.get_or_create(self._attractor_name)

    @property
    def is_star(self):
        return self.attractor is None

    @property
    def krpc_body(self):
        return UTIL_CONN.space_center.bodies[self.name]
        
    def orbit(self, epoch):
        from .orbit import Orbit
        return Orbit.from_krpcorb(self.krpc_body.orbit).propagate_to_epoch(epoch)
