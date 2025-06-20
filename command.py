from __future__ import annotations
from docopt import docopt
import random

from .astro.orbit import Orbit
from .astro.body import Body
from .task.launch import *
from .task.tasks import *
from .spacecrafts import *
from .utils import *


LAUNCH_DOC = f"""
Usage:
    ! launch [options] [<elements> ...]
    ! l [options] [<elements> ...]

"!launch", "!l"指令用于发射新载具.

如: "!launch -o 250 -i 28.61"(将默认火箭发射到倾角28.61°, 250km圆轨),
    "!launch -r soyuz2 -p r"(发射搭载中继卫星的联盟2号运载火箭到默认轨道),
    "!launch -e 7000 0.01 45 80 30 60"(发射到轨道根数确定的目标轨道).

Options:
    -h --help               查看发射任务指令帮助.
    -b --body <b>           目标天体, 使用"!bodies"指令查看详情. [default: earth]
    -n --name <n>           载荷名称.
    -r --rocket <r>         运载火箭, 使用"!rkt"指令查看详情. [default: soyuz2]
    -p --payload <p>        载荷, 使用"!payload"指令查看详情. [default: r]
    -c --circular <o>       圆轨道高度, 启用此参数将覆盖近/远拱点参数.
    -a --apoapsis <ap>      远拱点(km).
    -e --periapsis <pe>     近拱点(km).
    -i --inclination <i>    倾角(deg). [default: 45]
    -w --priority <w>       优先级, 0: 常规, 1: 高优先级, 2: 紧急. [default: 0]
    -t --time <t>           点火时(请用"T"替换空格, 支持大多数时间格式,
                            如19510107T105859, 或10:58:59(当天)).
    -m --elements           使用轨道根数确定目标轨道,
                            半长轴(km), 离心率(1), 倾角(deg), 
                            升交点赤经(deg), 近地点辐角(deg), 真近点角(deg).
"""

STATION_DOC = f"""
Usage:
    ! <station> [options]

"!<station>"指令用于部署空间站/地面站任务.
使用"!station"指令查看可用空间站/地面站详情.

如: "!kss -c"(kss空间站乘员任务),
    "!kss --supply"(kss空间站补给任务),

Options:
    -h --help               查看空间站任务指令帮助.
    -c --crew               部署乘员任务.
    -s --supply             部署补给任务.
"""

BODY_DOC = f"""
Usage:
    ! <body> [options] [<elements> ...]

"!<body>"指令用于部署行星/卫星着陆任务.
使用"!bodies"指令查看太阳系与支持的天体.
使用"!<body> -h"指令(如"!moon -h")查看天体详情与支持的任务.
如: "!moon -s trq"(着陆到月球静海站)
    "!moon -m 50 -30"(着陆到月球(50°N, 30°W))

Options:
    -h --help               查看天体详情.
    -s --site               着陆场.
    -m --elements           着陆场的坐标.
"""


class ChatMsg:
    def __init__(self,
                 chat_id: str,
                 chat_text: str,
                 user_id: str,
                 user_name: str,
                 time: int):
        self.chat_id = chat_id
        self.chat_text = chat_text
        self.user_id = user_id
        self.user_name = user_name
        self.time = time

    def __str__(self):
        return f'{self.user_name}: {self.chat_text} @ {self.time}'

    def _to_dict(self):
        return {
            'chat_id': self.chat_id,
            'chat_text': self.chat_text,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'time': self.time
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(data['chat_id'], data['chat_text'], data['user_id'], data['user_name'], data['time'])


class Command:
    _lastest_body_mission_dic = {}

    def __init__(self, msg: ChatMsg):
        self.tasks: Tasks | None = None
        self.importance = 0
        self.start_time = -1
        self.msg = msg

    def process(self) -> Tasks | None:
        """
        处理命令并返回生成的Tasks
        """
        command_args = self.msg.chat_text[1:].split()
        command_type = command_args[0]
        if command_type in ['rocket', 'rkt']:
            # TODO:
            return
        if command_type == 'payload':
            return
        if command_type == 'remove':
            TaskQueue.remove_by_user(self.msg.user_id)
            return
        if command_type == 'queue':
            return
        if command_type in ['launch', 'l']:
            return self._launch_command_process(command_args)
        if command_type in SPACESTATION_DIC.keys():
            return self._spacestation_command_process(command_args)
        if command_type == 'station':
            return
        if command_type == 'n':
            return
        LOGGER.warning(f'@{self.msg.user_id} 未知指令{command_type}, 使用"!help"指令查看可用指令.')

    def _launch_command_process(self, command_args) -> Tasks | None:
        try: 
            args = docopt(LAUNCH_DOC, command_args)
        except SystemExit:
            if '-h' in command_args or '--help' in command_args:
                # TODO:
                return
            LOGGER.warning(f'@{self.msg.user_id} 无法解析指令, 使用"!launch -h"指令查看发射指令帮助.')
            return

        rocket = args['--rocket']
        executor = LAUNCH_ROCKET_DIC.get(rocket, None)
        if not executor:
            LOGGER.warning(f'@{self.msg.user_id} 未知运载火箭{rocket}, 使用"!rkt"指令查看可用运载火箭.')
            return

        payload = LAUNCH_PAYLOAD_DIC.get(args['--payload'], None)
        if not payload:
            LOGGER.warning(f'@{self.msg.user_id} 未知载荷{payload}, 使用"!payload"指令查看可用载荷.')
            return
        if payload not in executor.payload_type:
            LOGGER.warning(f'@{self.msg.user_id} {rocket}不能搭载{payload}, 使用“!rocket”指令查看运载火箭可用载荷.')
            return

        if args['--elements']:
            a, e, inc, raan, argp, nu = args['<elements>']
            a = a * 1000
            inc = np.deg2rad(inc)
            raan = np.deg2rad(raan)
            argp = np.deg2rad(argp)
            nu = np.deg2rad(nu)
        else:
            pe = args['--periapsis']
            ap = args['--apoapsis']
            c = args['--circular']
            inc = args['--inclination']
            if ap and pe:
                ap = float(ap) * 1000
                pe = float(pe) * 1000
            elif ap:
                ap = float(ap) * 1000
                pe = 200000.
            elif c:
                ap, pe = float(c) * 1000., float(c) * 1000.
            else:
                ap, pe = 220000., 200000.
            ap, pe = max(ap, pe), min(ap, pe)
            ap, pe = max(ap, 200000.0), max(pe, 200000.0)
            earth = Body.get_or_create('Earth')
            ra, rp = ap + earth.r, pe + earth.r
            a = (ra + rp) / 2
            e = (ra - rp) / (ra + rp)
            inc = np.deg2rad(inc)
            raan = random.random() * 2 * np.pi
            argp = 0
            nu = 0
        orbit = Orbit.from_coe(earth, a, e, inc, raan, argp, nu, 0)

        priority = args['--priority']
        start_time = date_to_sec(args['--time']) if args['--time'] else -1

        name = args['--name'] if args['--name'] else (f'{self.msg.user_name}的载荷')
        # 最后再处理name防止内存泄漏
        name = Spacecraft(name).name

        self.tasks = Tasks(self.msg)
        self.tasks.submit(executor(
            spacecraft=Spacecraft(name), 
            tasks=self.tasks, 
            orbit=orbit,
            payload=payload, 
            start_time=start_time,
            importance=int(priority)))
        return self.tasks

    def _spacestation_command_process(self, command_args) -> Tasks | None:
        try: 
            args = docopt(STATION_DOC, command_args)
        except SystemExit:
            if '-h' in command_args or '--help' in command_args:
                return
            LOGGER.warning(f'@{self.msg.user_id} 无法解析指令, 使用"!ss -h"指令查看空间站指令帮助.')
            return

        self.tasks = Tasks(self.msg)
        spacestation = SPACESTATION_DIC.get(args['<station>'], None)
        if not spacestation:
            LOGGER.warning(f'@{self.msg.user_name} 未知空间站{args["<station>"]}, 使用"!ss"指令查看可用空间站.')
            return

        if args['--supply']:
            task_list = spacestation.supply_mission(self.tasks)
            self.tasks.submit_nowait(task_list)
            return self.tasks
        elif args['--crew']:
            task_list = spacestation.crew_mission(self.tasks)
            self.tasks.submit_nowait(task_list)
            return self.tasks

