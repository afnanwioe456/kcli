from __future__ import annotations

import argparse
import random

from .task.launch import *
from .task.tasks import Tasks
from .spacecrafts import *
from .utils import *
from .natural_command_process import natural_command

if TYPE_CHECKING:
    from task.tasks import TaskQueue, Task


class NoValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # if values is not None:
        #     print(f"参数 {option_string} 忽略参数 {values}")
        setattr(namespace, self.dest, True)


class SingleArgumentAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # if getattr(namespace, self.dest, None) is not None:
        #     print(f"参数 {option_string} 只能接受一个参数")
        setattr(namespace, self.dest, values)


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


class Command:
    count = 0

    def __init__(self, msg: ChatMsg):
        self.tasks: Tasks | None = None
        self.importance = 0
        self.start_time = -1
        self.msg = msg
        Command.count += 1
        self.count = Command.count

    def process(self, task_queue: TaskQueue) -> Tasks | None:
        """
        处理命令并返回生成的Tasks
        """
        # 将命令行字符串转为参数列表
        command_args = self.msg.chat_text.lower().split()
        command_type = command_args[0]
        if command_type[1:] in ['launch', 'la']:
            return self._launch_command_process(command_args, task_queue)
        elif command_type[1:] in SPACESTATION_NAME_DIC.keys():
            return self._spacestation_command_process(command_args, task_queue)
        elif command_type == '!n':
            try:
                res = natural_command(self.msg.chat_text)
                if res is None:
                    return
                self.msg.chat_text = res
                return self.process(task_queue)
            except Exception as e:
                LOGGER.exception('natural command process error', exc_info=e)
                return
        else:
            LOGGER.warning(f'@{self.msg.user_id} 指令不存在！发送 !help 查看可用指令')
            return

    def _launch_command_process(self, command_args, task_queue) -> Tasks | None:
        apr = argparse.ArgumentParser(
            description='"!launch"指令用于发射新载具，'
                        '如："!launch -o 250 -i 28.61"(将默认火箭发射到倾角28.61°, 250km圆轨) '
                        '"!launch -r soyuz2 -p r"(发射搭载中继卫星的联盟2号运载火箭到默认250km轨道)。'
                        '可选参数如下：')

        apr.add_argument('-n', '--name', type=str, help='载荷的名称', metavar='')
        apr.add_argument('-r', '--rocket', type=str, help='运载火箭,发送"!rocket"查看详情', metavar='')
        apr.add_argument('-p', '--payload', type=str, help='载荷,发送"!payload"查看详情', metavar='')
        apr.add_argument('-o', '--orbit', type=float, help='轨道的高度(默认圆轨)', metavar='')
        apr.add_argument('-ap', '--apoapsis', type=float, help='远拱点(km)', metavar='')
        apr.add_argument('-pe', '--periapsis', type=float, help='近拱点(km)', metavar='')
        apr.add_argument('-i', '--inclination', type=float, help='轨道倾角', metavar='')
        apr.add_argument('-w', '--priority', type=int, help='优先级 0：常规 1：高优先级 2：紧急', metavar='')
        apr.add_argument('-t', '--time', type=str, help='点火时(请用"T"替换空格,支持大多数时间格式,'
                                                        '如19510107T105859,或10:58:59(当天))', metavar='')

        # 解析命令行参数
        try:
            args = apr.parse_args(command_args[1:])
        except Exception as e:
            LOGGER.exception('Error in launch command process', exc_info=e)
            return

        self.name = args.name if args.name else (self.msg.user_name + '的载荷')
        self.name = vessel_namer(self.name)

        rocket = args.rocket if args.rocket else random.choice(list(LAUNCH_ROCKET_DIC.keys()))
        if rocket not in LAUNCH_ROCKET_DIC.keys():
            LOGGER.warning(f'@{self.msg.user_id} 运载火箭不存在！输入"!rkt"查看可用运载火箭类型...')
            return
        executor = LAUNCH_ROCKET_DIC[rocket]

        self.payload = args.payload if args.payload else 'r'
        if self.payload not in LAUNCH_PAYLOAD_DIC.keys():
            LOGGER.warning(f'@{self.msg.user_id} 载荷不存在！输入"!payload"查看可用载荷类型...')
            return
        self.payload = LAUNCH_PAYLOAD_DIC[self.payload]
        if self.payload not in executor.payload_type:
            LOGGER.warning(f'@{self.msg.user_id} 该运载火箭不能搭载该载荷！输入“!rocket”查看运载火箭可用载荷...')
            return

        if args.apoapsis and args.periapsis:
            self.ap = args.apoapsis * 1000.0
            self.pe = args.periapsis * 1000.0
        elif args.apoapsis:
            self.ap = args.apoapsis * 1000.0
            self.pe = 200000.0
        elif args.orbit:
            self.ap, self.pe = args.orbit * 1000.0, args.orbit * 1000.0
        else:
            self.ap, self.pe = 250000.0, 200000.0
        # TODO: 如果pe大于一定高度的话，submit发射+变轨

        self.ap, self.pe = max(self.ap, self.pe), min(self.ap, self.pe)
        self.ap, self.pe = max(self.ap, 200000.0), max(self.pe, 200000.0)

        self.inc = args.inclination if args.inclination is not None else 50.0

        if args.priority == 0:
            self.priority = 1
        elif args.priority == 1:
            self.priority = 2
        else:
            self.priority = 3

        start_sec = date_to_sec(args.time)
        self.start_time = start_sec if start_sec else -1

        self.tasks = Tasks(self.msg, self.count, task_queue)
        self.tasks.submit(executor(tasks=self.tasks, name=self.name, rocket_name=executor.rocket_name,
                                   payload_name=self.payload, ap_altitude=self.ap, pe_altitude=self.pe,
                                   inclination=self.inc, start_time=self.start_time))
        return self.tasks

    def _spacestation_command_process(self, command_args, task_queue) -> Tasks | None:
        apr = argparse.ArgumentParser(
            description='"!空间站"指令用于部署空间站任务(发送"!ss"查看空间站代码)，'
                        '如："!kss -s"(近地空间站补给任务)。'
                        '可选参数如下：')

        apr.add_argument('-s', '--supply', nargs='?', const=True, default=False, action=NoValueAction,
                         help='补给任务', metavar='')
        apr.add_argument('-c', '--crew', nargs='?', const=True, default=False, action=NoValueAction,
                         help='成员任务', metavar='')
        apr.add_argument('-w', '--priority', type=int, help='优先级 0：常规 1：高优先级 2：紧急', metavar='')

        try:
            args = apr.parse_args(command_args[1:])
        except Exception as e:
            LOGGER.exception('Error in spacestation command process', exc_info=e)
            return

        if args.priority == 0:
            self.priority = 1
        elif args.priority == 1:
            self.priority = 2
        else:
            self.priority = 3

        self.tasks = Tasks(self.msg, self.count, task_queue)
        spacestation: Spacestation = SPACESTATION_DIC[command_args[0][1:]]

        task = None
        if args.supply:
            task = spacestation.supply_mission(self.tasks)
        elif args.crew:
            task = spacestation.crew_mission(self.tasks)
        if task:
            self.tasks.submit(task)
            return self.tasks


class ShortCommand(Command):
    def __init__(self, msg: ChatMsg):
        super().__init__(msg)

    def process(self, task_queue: TaskQueue) -> Tasks | None:
        command_args = self.msg.chat_text.lower().split()
        command_type = command_args[0]
        if self.msg.chat_text == '!launch -h':
            write_helper('launch')
            # TODO: 异步更新
            return
        elif self.msg.chat_text == '!help':
            write_helper('command')
            return

        match command_type:
            case '!rkt':
                return _rocket_help()
            case '!payload':
                return _payload_help()
            case '!queue':
                print(task_queue)
                return
            case '!remove':
                task_queue.remove_by_user(self.msg.user_name)
                return
            case '!time':
                print(f'KSP世界时：{sec_to_date(int(get_ut()))} UTC')
            case '!launch':
                return super().process(task_queue)
            case '!la':
                return super().process(task_queue)
            case '!n':
                return super().process(task_queue)
            # TODO: abort, save

        if command_type[1:] in SPACESTATION_DIC.keys():
            return super().process(task_queue)


def _rocket_help():
    print(LINE_SEP)
    print('这是一条临时rocket帮助，维护中...')
    print('+ 可用运载火箭：\n'
          '-r soyuz2        联盟2号\n'
          '-r ariane5       阿丽亚娜5型\n'
          '-r cz7           长征7号\n'
          '如：!launch -r 运载火箭 -o 450 ...')


def _payload_help():
    print('这是一条rocket帮助, 当前请忽略-p参数使用默认载荷')
