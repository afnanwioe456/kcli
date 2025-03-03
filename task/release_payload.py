from __future__ import annotations
from time import sleep
from krpc.services.spacecenter import Vessel, VesselType
from astropy import units as u

from .tasks import Task
from ..utils import *

if TYPE_CHECKING:
    from .tasks import Tasks
    from ..spacecrafts import Spacecraft

__all__ = [
    'ReleasePayload',
]


class ReleasePayload(Task):
    def __init__(self,
                 spacecraft: Spacecraft,
                 tasks: Tasks,
                 start_time: u.Quantity = -1 * u.s,
                 duration: u.Quantity = 180 * u.s,
                 importance: int = 0,
                 count: int = 1,  # 待分离的载荷数
                 ):
        super().__init__(spacecraft, tasks, start_time, duration, importance)
        self.count = count
        self.original_name = self.spacecraft._original_name

    @property
    def description(self):
        return (f'{self.name} 释放载荷: {self.count}\n'
                f'\t预计执行时: {sec_to_date(self.start_time)}')

    @staticmethod
    def _deploy(vessel: Vessel,
                wait=5,
                antenna=True,
                solar_panel=True,
                ):
        sleep(wait)
        if antenna:
            deploy_antenna(vessel)
        if solar_panel:
            deploy_solar_panels(vessel)

    @logging_around
    def start(self):
        # TODO: 新载荷spacecraft对象问题
        if not self._conn_setup():
            return
        self.vessel.name = vessel_namer(f'{self.name}_tmp')
        # 分级，释放第一个载荷会自动切换到载荷上
        self.vessel.control.activate_next_stage()
        LOGGER.debug(f'释放载荷: {self.vessel.name}')
        self.count -= 1
        adapter_name = self.vessel.name
        # 将active_vessel切换到释放的载具上并命名
        self.vessel = self.sc.active_vessel
        self.vessel.name = self.name
        self._deploy(self.vessel)

        # 尝试切换到适配器上
        if switch_to_vessel(adapter_name):
            self.vessel = self.sc.active_vessel
        elif self.conn is not None:
            self.conn.close()
            return
        else:
            return

        # 如果还有载荷
        while self.count > 0:
            past_vessels = self.sc.vessels
            # 分级，注意ksp不会自动切换到载荷上
            self.vessel.control.activate_next_stage()
            current_vessels = self.sc.vessels
            new_payloads = get_new_vessels(past_vessels, current_vessels)
            for p in new_payloads:
                p.name = vessel_namer(self.original_name)
                self._deploy(p, 3)
                LOGGER.debug(f'释放载荷: {p.name}')
            sleep(5)
            self.count -= 1

        self.vessel.type = VesselType.debris
        self.vessel.name += ' Debris'
        self.conn.close()

    def _to_dict(self):
        dic = {
            'count': self.count,
        }
        return super()._to_dict() | dic
    
    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        return cls(
            spacecraft = SpacecraftBase.get(data['spacecraft_name']),
            tasks = tasks,
            start_time = data['start_time'] * u.s,
            duration = data['duration'] * u.s,
            importance = data['importance'],
            count = data['count'],
        )


def deploy_solar_panels(vessel: Vessel):
    solar_panels = vessel.parts.solar_panels
    for sp in solar_panels:
        if sp.deployable:
            sp.deployed = True


def deploy_antenna(vessel: Vessel):
    antennas = vessel.parts.antennas
    for a in antennas:
        if a.deployable:
            a.deployed = True
