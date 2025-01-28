from __future__ import annotations
from time import sleep
import krpc

from .tasks import Task
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft

__all__ = [
    'Launch',
    'Soyuz2Launch',
    'LongMarch7Launch',
    'LAUNCH_ROCKET_DIC',
    'LAUNCH_PAYLOAD_DIC',
]

class Launch(Task):
    payload_type = []  # 可以搭载的载荷
    rocket_name = "Undefined"

    def __init__(self,
                 spacecraft: Spacecraft,
                 tasks: Tasks,
                 start_time: int = -1,
                 duration: int = 3600,
                 importance: int = 3,
                 payload: str = "Relay",
                 crew_name_list: None | list = None,
                 path_index: int = 2,
                 pe_altitude: float = 200000,
                 ap_altitude: float = 250000,
                 inclination: float = 19.61,
                 autostage: bool = True,
                 ):
        super().__init__(spacecraft, tasks, start_time, duration, importance)

        self.payload = payload
        if crew_name_list is None:
            self.crew_name_list = []
        else:
            self.crew_name_list = crew_name_list

        self.path_index = path_index
        self.pe_altitude = pe_altitude
        self.pe_desired = None
        self.inclination = inclination
        self.inc_desired = None
        self.ap_altitude = ap_altitude
        self.autostage = True

    @property
    def description(self):
        return (
            f'{self.name} -> 发射\n'
            f'\t运载火箭: {self.rocket_name} 载荷: {self.payload}\n'
            f'\t近拱点: {self.pe_altitude / 1000}km 远拱点: {self.ap_altitude / 1000}km\n'
            f'\t倾角: {self.inclination}\n'
            f'\t预计点火时: {sec_to_date(self.start_time)}')

    def _conn_setup(self):
        setup_flag = super()._conn_setup()
        if not setup_flag or self.vessel is None:
            return False
        self.vessel.name = self.name
        self.autopilot = self.mj.ascent_autopilot
        self.autopilot_path_index = self.path_index
        if self.autopilot_path_index == 2:
            self.pvg_ascent = self.autopilot.ascent_path_pvg
        # TODO
        self.pvg_ascent = self.autopilot.ascent_path_pvg

        if self.ap_altitude == self.pe_altitude:
            self.pe_altitude -= 100
        if self.pe_altitude > 400000:
            self.pe_desired = self.pe_altitude
            self.pe_altitude = 400000
        if 19.61 > self.inclination > 0:
            self.inc_desired = self.inclination
            self.inclination = 19.61
        elif -19.61 < self.inclination < 0:
            self.inc_desired = self.inclination
            self.inclination = -19.61

        self.autopilot.desired_orbit_altitude = self.pe_altitude
        self.autopilot.desired_inclination = self.inclination
        self.pvg_ascent.desired_apoapsis = self.ap_altitude
        # self.pvg_ascent.desired_attach_alt = self.pe_altitude
        # self.pvg_ascent.attach_alt_flag = True

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
            LOGGER.debug(f'{self.name}: rolling out')
            sc.launch_vessel('VAB', f'{self.rocket_name}_{self.payload}', 'LaunchPad', True, [])
            sc.active_vessel.name = self.name

    @logging_around
    def start(self):
        self._roll_out()
        if not self._conn_setup():
            return
        time_wrap(self.start_time)
        self.autopilot.enabled = True
        LOGGER.debug(f'{self.name}: autopilot engaged')
        while not self.tasks.abort_flag and self.autopilot.enabled:
            sleep(10)
        self._submit_next_task()
        self.conn.close()

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
            'payload': self.payload,
            'crew_name_list': self.crew_name_list,
            'path_index': self.path_index,
            'pe_altitude': self.pe_altitude,
            'ap_altitude': self.ap_altitude,
            'inclination': self.inclination,
            'autostage': self.autostage,
        }
        return super()._to_dict() | dic

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        return cls(
            spacecraft = SpacecraftBase.get(data['spacecraft_name']),
            tasks = tasks,
            start_time = data['start_time'],
            duration = data['duration'],
            importance = data['importance'],
            payload = data['payload'],
            crew_name_list = data['crew_name_list'],
            path_index = data['path_index'],
            pe_altitude = data['pe_altitude'],
            ap_altitude = data['ap_altitude'],
            inclination = data['inclination'],
            autostage = data['autostage'],
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
        p_release_1 = ReleasePayload(self.spacecraft, self.tasks, count=2)
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
    rocket_name = 'LongMarch_7'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


LAUNCH_ROCKET_DIC: dict[str, Type[Launch]] = {
    'soyuz2': Soyuz2Launch,
    'ariane5': Ariane5ECALaunch,
    'cz7': LongMarch7Launch,
}

LAUNCH_PAYLOAD_DIC = {
    'r': 'Relay',
    'sysc': 'Soyuz_spacecraft'
}
