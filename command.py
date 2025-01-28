from __future__ import annotations

from docopt import docopt
import random

from .task.launch import *
from .task.tasks import Tasks
from .spacecrafts import *
from .utils import *

if TYPE_CHECKING:
    from task.tasks import TaskQueue

LAUNCH_DOC = f"""
Usage:
    ! launch [options] [<elements> ...]
    ! la [options] [<elements> ...]

"!launch"指令用于发射新载具,
如: "!launch -o 250 -i 28.61"(将默认火箭发射到倾角28.61°, 250km圆轨),
    "!launch -r soyuz2 -p r"(发射搭载中继卫星的联盟2号运载火箭到默认轨道),
    "!launch -e 7000 0.01 45 80 30 60"(发射到轨道根数确定的目标轨道).

Options:
    -h --help               查看发射任务指令帮助.
    -n --name <n>           载荷名称.
    -r --rocket <r>         运载火箭, 使用"!rkt"指令查看详情. [default: soyuz2]
    -p --payload <p>        载荷, 使用"!payload"指令查看详情. [default: r]
    -o --orbit <o>          圆轨道高度, 启用此参数将覆盖近/远拱点参数.
    -a --apoapsis <ap>      远拱点(km).
    -e --periapsis <pe>     近拱点(km).
    -i --inclination <i>    倾角(deg). [default: 45]
    -w --priority <w>       优先级, 0: 常规, 1: 高优先级, 2: 紧急. [default: 0]
    -t --time <t>           点火时(请用"T"替换空格, 支持大多数时间格式,
                            如19510107T105859, 或10:58:59(当天)).
    -m --elements           使用轨道根数确定目标轨道,
                            半长轴(km), 离心率(1), 倾角(deg), 升交点赤经(deg), 近地点辐角(deg), 真近点角(deg).
"""

SS_DOC = f"""
Usage:
    ! <space-station> -h
    ! <space-station> (-c | -s) [options]

"!<space-station>"指令用于部署空间站任务, 使用"!ss"指令查看可用空间站详情.
如: "!kss -c"(kss空间站乘员任务),
    "!kss --supply"(kss空间站补给任务),

Options:
    -h --help               查看空间站任务指令帮助.
    -c --crew               部署乘员任务.
    -s --supply             部署补给任务.
"""

ROCKET_DOC = f"""
这是一条临时rocket帮助, 维护中...
可用运载火箭:
soyuz2        联盟2号
ariane5       阿丽亚娜5型
cz7           长征7号
如：!launch -r 运载火箭 -o 450
"""

PAYLOAD_DOC = f"""
这是一条临时payload帮助, 当前请忽略-p参数使用默认载荷
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
    def __init__(self, msg: ChatMsg):
        self.tasks: Tasks | None = None
        self.importance = 0
        self.start_time = -1
        self.msg = msg

    def process(self, task_queue: TaskQueue) -> Tasks | None:
        """
        处理命令并返回生成的Tasks
        """
        command_args = self.msg.chat_text[1:].lower().split()
        command_type = command_args[0]
        if command_type in ['rocket', 'rkt']:
            # TODO:
            return
        if command_type == 'payload':
            return
        if command_type == 'remove':
            task_queue.remove_by_user(self.msg.user_id)
            return
        if command_type == 'queue':
            return
        if command_type in ['launch', 'la']:
            return self._launch_command_process(command_args, task_queue)
        if command_type in SPACESTATION_DIC.keys():
            return self._spacestation_command_process(command_args, task_queue)
        if command_type == 'ss':
            return
        if command_type == 'n':
            return
        LOGGER.warning(f'@{self.msg.user_id} 未知指令{command_type}, 使用"!help"指令查看可用指令.')

    def _launch_command_process(self, command_args, task_queue) -> Tasks | None:
        try: 
            args = docopt(LAUNCH_DOC, command_args)
        except SystemExit:
            if '-h' in command_args or '--help' in command_args:
                # TODO:
                return
            LOGGER.warning(f'@{self.msg.user_id} 无法解析指令, 使用"!launch -h"指令查看发射指令帮助.')
            return

        name = args['--name'] if args['--name'] else (f'{self.msg.user_name}的载荷')
        name = vessel_namer(name)

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

        pe, ap, o = args['--periapsis'], args['--apoapsis'], args['--orbit']
        if ap and pe:
            ap = float(ap) * 1000.0
            pe = float(pe) * 1000.0
        elif ap:
            ap = float(ap) * 1000.0
            pe = 200000.0
        elif o:
            ap, pe = float(o) * 1000.0, float(o) * 1000.0
        else:
            ap, pe = 250000.0, 200000.0
        # TODO: 如果pe大于一定高度的话，submit发射+变轨
        # TODO: 轨道根数发射
        ap, pe = max(ap, pe), min(ap, pe)
        ap, pe = max(ap, 200000.0), max(pe, 200000.0)

        inc = args['--inclination']
        priority = args['--priority']
        start_time = date_to_sec(args['--time']) if args['--time'] else -1

        self.tasks = Tasks(self.msg, task_queue)
        self.tasks.submit(executor(
            tasks=self.tasks, 
            spacecraft=Spacecraft(name), 
            payload=payload, 
            ap_altitude=ap, 
            pe_altitude=pe,
            inclination=inc, 
            start_time=start_time,
            importance=int(priority)))
        return self.tasks

    def _spacestation_command_process(self, command_args, task_queue) -> Tasks | None:
        try: 
            args = docopt(SS_DOC, command_args)
        except SystemExit:
            if '-h' in command_args or '--help' in command_args:
                return
            LOGGER.warning(f'@{self.msg.user_id} 无法解析指令, 使用"!ss -h"指令查看空间站指令帮助.')
            return

        self.tasks = Tasks(self.msg, task_queue)
        spacestation = SPACESTATION_DIC.get(args['<space-station>'], None)
        if not spacestation:
            LOGGER.warning(f'@{self.msg.user_name} 未知空间站{args["<space-station>"]}, 使用"!ss"指令查看可用空间站.')
            return

        if args['--supply']:
            task_list = spacestation.supply_mission(self.tasks)
            self.tasks.submit_nowait(task_list)
            return self.tasks
        elif args['--crew']:
            task_list = spacestation.crew_mission(self.tasks)
            self.tasks.submit_nowait(task_list)
            return self.tasks

