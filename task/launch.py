from __future__ import annotations
from time import sleep
import krpc
from krpc.client import Client
from krpc.services.spacecenter import Vessel

from .tasks import Task
from ..kerbals import Kerbal
from ..utils import *

if TYPE_CHECKING:
    from task.tasks import Tasks


class Launch(Task):
    payload_type = []  # 可以搭载的载荷
    rocket_name = "Soyuz_2"
    payload_name = "_Relay"

    def __init__(self,
                 tasks: Tasks,
                 name: str = "test flight",
                 rocket_name: str = "Soyuz_2",
                 payload_name: str = "_Relay",
                 crew_name_list: None | list = None,
                 path_index: int = 2,
                 pe_altitude: float = 200000,
                 ap_altitude: float = 250000,
                 inclination: float = 19.61,
                 start_time: int = -1,  # 执行时间，传递给任务管理器修正
                 duration: int = 3600,
                 importance: int = 3,
                 autostage: bool = True,
                 ):
        super().__init__(name, tasks, start_time, duration, importance)

        self.conn:  Client | None = None  # 连接尚未建立
        self.vessel: Vessel | None = None  # 火箭尚未推出
        self.rocket_name = rocket_name
        self.payload_name = payload_name
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
        self.autostage = autostage

        self.time_out = 3600

    @property
    def description(self):
        return (
            f"{self.name}->发射\n"
            f"\t运载火箭: {self.rocket_name} 载荷: {self.payload_name[1:]}\n"
            f"\t近拱点: {self.pe_altitude / 1000}km 远拱点: {self.ap_altitude / 1000}km\n"
            f"\t倾角: {self.inclination}\n"
            f"\t预计点火时: {'ASAP' if self.start_time <= 0 else sec_to_date(self.start_time)}")

    def _conn_setup(self, **kwargs):
        setup_flag = super()._conn_setup(f"launch: {self.name}")
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
        self.autopilot.enabled = True
        LOGGER.debug(f'{self.name}: autopilot engaged')

    def _roll_out(self):  # 推出火箭
        with krpc.connect(name=self.rocket_name + ' roll out') as conn:
            sc = conn.space_center
            if sc is None:
                return
            LOGGER.debug(f'{self.name}: rolling out...')
            sc.launch_vessel('VAB', self.rocket_name + self.payload_name, 'LaunchPad', True, [])

    @logging_around
    def start(self):
        self._roll_out()
        self._conn_setup()
        time_wrap(self.start_time)
        self._activate_next_stage()
        while not self.tasks.abort_flag and self.autopilot.enabled:
            sleep(10)
        self._submit_next_task()
        self.conn.close()

    def _submit_next_task(self):
        from .maneuver import SimpleMnv
        from .release_payload import ReleasePayload
        new_task = []
        payload_release = ReleasePayload(self.name, self.tasks)
        new_task.append(payload_release)
        if self.pe_desired is not None:
            pe_mnv = SimpleMnv(self.name, self.tasks, 'pe', self.pe_desired)
            new_task.append(pe_mnv)
        if self.inc_desired is not None:
            inc_mnv = SimpleMnv(self.name, self.tasks, 'inc', self.inc_desired)
            new_task.append(inc_mnv)
        self.tasks.submit_nowait(new_task)


class Soyuz2Launch(Launch):
    payload_type = ['_Relay', '_Soyuz_spacecraft']
    rocket_name = 'Soyuz_2'

    def __init__(self, tasks, **kwargs):
        super().__init__(tasks, **kwargs)

class Ariane5ECALaunch(Launch):
    payload_type = ['_Relay']
    rocket_name = 'Ariane_5_ECA'

    def __init__(self, tasks, **kwargs):
        super().__init__(tasks, **kwargs)

    def _submit_next_task(self):
        from .maneuver import SimpleMnv
        from .release_payload import ReleasePayload
        new_task = []
        p_release_1 = ReleasePayload(self.name, self.tasks, count=2)
        p_release_2 = ReleasePayload(self.name, self.tasks)

        new_task.append(p_release_1)
        if self.pe_desired is not None:
            pe_mnv = SimpleMnv(self.name, self.tasks, 'pe', self.pe_desired)
            new_task.append(pe_mnv)
        if self.inc_desired is not None:
            inc_mnv = SimpleMnv(self.name, self.tasks, 'inc', self.inc_desired)
            new_task.append(inc_mnv)
        new_task.append(p_release_2)
        self.tasks.submit_nowait(new_task)


class LongMarch7Launch(Launch):
    payload_type = ['_Relay']
    rocket_name = 'LongMarch_7'

    def __init__(self, tasks, **kwargs):
        super().__init__(tasks, **kwargs)


LAUNCH_ROCKET_DIC: dict[str, Type[Launch]] = {
    'soyuz2': Soyuz2Launch,
    'ariane5': Ariane5ECALaunch,
    # 'cz7': la.LongMarch7Launch,
}
LAUNCH_PAYLOAD_DIC = {
    'r': '_Relay',
    'sysc': '_Soyuz_spacecraft'
}
