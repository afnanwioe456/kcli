from __future__ import annotations
from time import sleep
from typing import Type
import numpy as np
import krpc

from .tasks import Task
from ..astro.frame import BCIFrame
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..astro.orbit import Orbit
    from ..spacecrafts import Spacecraft

__all__ = [
    'Launch',
    'Soyuz2Launch',
    'Ariane5ECALaunch',
    'LongMarch7Launch',
    'LongMarch10Launch',
    'LAUNCH_ROCKET_DIC',
    'LAUNCH_PAYLOAD_DIC',
]

class Launch(Task):
    payload_type = []  # 可以搭载的载荷
    rocket_name = "Undefined"

    def __init__(self,
                 spacecraft: Spacecraft,
                 tasks: Tasks,
                 orbit: Orbit,
                 payload: str = "Relay",
                 crew_name_list: None | list[str] = None,
                 direction: str = 'SE',
                 start_time: float = -1,
                 duration: float = 1800,
                 importance: int = 3,
                 submit_next: bool = True):
        super().__init__(spacecraft, tasks, start_time, duration, importance, submit_next)
        self.orbit = orbit
        self.direction = direction
        self.start_time = max(UTIL_CONN.space_center.ut, start_time)
        site = LAUNCH_SITES_COORDS['wenchang']
        launch_window = self.orbit.launch_window(
            site_coord  = site, 
            direction   = self.direction, 
            cloest      = True,
            start_time  = self.start_time
        )
        # TODO
        self.start_time = launch_window - 360

        self.payload = payload
        if crew_name_list is None:
            self.crew_name_list = []
        else:
            self.crew_name_list = crew_name_list
        self.pe_altitude = self.orbit.pe
        self.pe_desired = None
        if self.direction in ['SE', 'SW']:
            self.inclination = -self.orbit.inc
        else:
            self.inclination = self.orbit.inc
        self.inc_desired = None
        self.ap_altitude = self.orbit.ap
        self.autostage = True

    @property
    def description(self):
        return (
            f'{self.spacecraft.name} -> 发射\n'
            f'\t运载火箭: {self.rocket_name} 载荷: {self.payload}\n'
            f'\t近拱点: {self.pe_altitude / 1000:.2f} 远拱点: {self.ap_altitude / 1000:.2f}\n'
            f'\t倾角: {np.rad2deg(self.inclination):.2f}\n'
            f'\t预计点火时: {sec_to_date(self.start_time)}'
        )

    def _conn_setup(self):
        setup_flag = super()._conn_setup()
        if not setup_flag or self.vessel is None:
            return False
        self.vessel.name = self.spacecraft.name
        self.autopilot = self.mj.ascent_autopilot
        self.pvg_ascent = self.autopilot.ascent_path_pvg

        if self.ap_altitude == self.pe_altitude:
            self.pe_altitude -= 100
        if self.pe_altitude > 400000:
            self.pe_desired = self.pe_altitude
            self.pe_altitude = 400000
        # TODO
        latitude = np.deg2rad(LAUNCH_SITES_COORDS['wenchang'][0])
        if abs(self.inclination) < abs(latitude):
            self.inc_desired = self.inclination
            self.inclination = abs(latitude) * self.inclination / abs(self.inclination)

        self.autopilot.desired_orbit_altitude = self.pe_altitude
        self.autopilot.desired_inclination = np.rad2deg(self.inclination)
        self.pvg_ascent.desired_apoapsis = self.ap_altitude

        self.autopilot.force_roll = True
        self.autopilot.vertical_roll = 90
        self.autopilot.turn_roll = 90
        self.autopilot.autostage = self.autostage
        return True

    def _roll_out(self):  # 推出火箭
        with krpc.connect(f'{self.rocket_name} roll out') as conn:
            sc = conn.space_center
            if sc is None:
                return
            LOGGER.debug(f'{self.spacecraft.name}: rolling out')
            sc.launch_vessel('VAB', f'{self.rocket_name}_{self.payload}', 'LaunchPad', True, [])
            sc.active_vessel.name = self.spacecraft.name

    @logging_around
    def start(self):
        self._roll_out()
        if not self._conn_setup():
            return
        time_wrap(self.start_time)
        self.autopilot.enabled = True
        LOGGER.debug(f'{self.spacecraft.name}: autopilot engaged')
        while not self.tasks.abort_flag and self.autopilot.enabled:
            sleep(10)
        if self.submit_next:
            self._submit_next_task()

    def _submit_next_task(self):
        from .maneuver import SimpleMnv
        from .release_payload import ReleasePayload
        new_task = []
        payload_release = ReleasePayload(self.spacecraft, self.tasks)
        new_task.append(payload_release)
        if self.pe_desired is not None:
            pe_mnv = SimpleMnv(self.spacecraft, self.tasks, 'pe', self.pe_desired)
            new_task.append(pe_mnv)
        if self.inc_desired is not None:
            inc_mnv = SimpleMnv(self.spacecraft, self.tasks, 'inc', self.inc_desired)
            new_task.append(inc_mnv)
        self.tasks.submit_nowait(new_task)

    def _to_dict(self):
        dic = {
            'orbit':            self.orbit._to_dict(),
            'payload':          self.payload,
            'crew_name_list':   self.crew_name_list,
            'direction':        self.direction
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import Spacecraft
        from ..astro.orbit import Orbit
        return cls(
            spacecraft      = Spacecraft.get(data['spacecraft_name']),
            tasks           = tasks,
            orbit           = Orbit._from_dict(data['orbit']),
            start_time      = data['start_time'],
            duration        = data['duration'],
            importance      = data['importance'],
            payload         = data['payload'],
            crew_name_list  = data['crew_name_list'],
            direction       = data['direction'],
            submit_next     = data['submit_next']
        )


class Soyuz2Launch(Launch):
    payload_type = ['Relay', 'Soyuz_spacecraft']
    rocket_name = 'Soyuz_2'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

class Ariane5ECALaunch(Launch):
    payload_type = ['Relay']
    rocket_name = 'Ariane_5_ECA'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _submit_next_task(self):
        from .maneuver import SimpleMnv
        from .release_payload import ReleasePayload
        new_task = []
        p_release_1 = ReleasePayload(self.spacecraft, self.tasks)
        p_release_2 = ReleasePayload(self.spacecraft, self.tasks)

        new_task.append(p_release_1)
        if self.pe_desired is not None:
            pe_mnv = SimpleMnv(self.spacecraft, self.tasks, 'pe', self.pe_desired)
            new_task.append(pe_mnv)
        if self.inc_desired is not None:
            inc_mnv = SimpleMnv(self.spacecraft, self.tasks, 'inc', self.inc_desired)
            new_task.append(inc_mnv)
        new_task.append(p_release_2)
        self.tasks.submit_nowait(new_task)


class LongMarch7Launch(Launch):
    payload_type = ['Relay']
    rocket_name = 'CZ_7'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LongMarch10Launch(Launch):
    payload_type = ['Lanyue']
    rocket_name = 'CZ_10'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _submit_next_task(self):
        from .maneuver import SimpleMnv
        new_task = []
        if self.pe_desired is not None:
            pe_mnv = SimpleMnv(self.spacecraft, self.tasks, 'pe', self.pe_desired)
            new_task.append(pe_mnv)
        if self.inc_desired is not None:
            inc_mnv = SimpleMnv(self.spacecraft, self.tasks, 'inc', self.inc_desired)
            new_task.append(inc_mnv)
        if new_task:
            self.tasks.submit_nowait(new_task)

LAUNCH_ROCKET_DIC: dict[str, Type[Launch]] = {
    'soyuz2':       Soyuz2Launch,
    'ariane5':      Ariane5ECALaunch,
    'cz7':          LongMarch7Launch,
    'cz10':         LongMarch10Launch,
}

LAUNCH_PAYLOAD_DIC = {
    'r':            'Relay',
    'sysc':         'Soyuz_spacecraft'
}
