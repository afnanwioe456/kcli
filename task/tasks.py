from __future__ import annotations
import krpc
import threading
from typing import TYPE_CHECKING

from ..utils import *

if TYPE_CHECKING:
    from command import ChatMsg
    from ..spacecrafts import SpacecraftBase
    
__all__ = [
    'MAX_IMPORTANCE',
    'Task',
    'Tasks',
    'TaskQueue',
]

MAX_IMPORTANCE = 9
"""
IMPORTANCE:
0: 弹幕普通发射任务 
1: 弹幕高优先级发射任务
2: 弹幕紧急发射任务
3: 对接
4: 
5: 默认优先级
6: 默认交会规划, 转移规划
7: 默认轨道机动
8: 交会机动
9: 捕获机动
"""


class Task:
    def __init__(self,
                 spacecraft: SpacecraftBase,
                 tasks: Tasks,
                 start_time: float,
                 duration: int,
                 importance: int,
                 ):
        self.spacecraft = spacecraft
        self.name = self.spacecraft.name
        self.tasks = tasks
        self.importance = importance
        self.start_time = start_time
        self.duration = duration

    @property
    def description(self) -> str:
        return f"undefined task"

    @property
    def short_description(self) -> str:
        return self.description.split('\n')[0]

    def reschedule(self, after_t):
        self.start_time = after_t

    def __str__(self):
        return self.description

    def _conn_setup(self):
        self.vessel = self.spacecraft.vessel
        if not switch_to_vessel(self.vessel.name):
            return False
        self.conn = krpc.connect(self.name)
        self.sc = self.conn.space_center
        if self.sc is None:
            return False
        self.mj = self.conn.mech_jeb # type: ignore
        return True

    @logging_around
    def start(self):
        self._conn_setup()

    def _to_dict(self):
        return {
            'type': self.__class__.__name__,
            'spacecraft_name': self.spacecraft.name,
            'start_time': self.start_time,
            'duration': self.duration,
            'importance': self.importance,
        }

    @classmethod
    def _from_dict(cls, data, tasks):
        from ..spacecrafts import SpacecraftBase
        s = SpacecraftBase.get(data['spacecraft_name'])
        return cls(s, tasks, data['start_time'], data['duration'], data['importance'])


class Tasks:
    _tasks_counter = 0

    def __init__(self, msg: ChatMsg):
        """
        任务序列，一个指令或某个任务自身产生的一系列任务
        """
        self.msg = msg
        self._task_list: list[Task] = []  # 等待执行的任务序列
        self.current_task: Task | None = None
        self._id = Tasks._tasks_counter
        Tasks._tasks_counter += 1
        self.abort_flag = False  # 某一个任务异常终止则放弃整个
    
    @property
    def next_task(self) -> Task | None:
        return self._task_list[0] if self._task_list else None

    def submit(self, task: Task | list[Task]):
        if not isinstance(task, list):
            task = [task]
        self._task_list += task

        log = (f"Tasks [{self._id}] added {len(task)} new Task(s):\n"
               f"{task[0].short_description}")
        LOGGER.debug(log)

    def submit_nowait(self, task: Task | list[Task]):
        if not isinstance(task, list):
            task = [task]
        self._task_list = task + self._task_list

        log = (f"Tasks [{self._id}] added {len(task)} new Task(s):\n"
               f"{task[0].short_description}")
        LOGGER.debug(log)

    def do(self):
        if not self.next_task:
            return
        self.current_task = self._task_list.pop(0)
        log = (f"Tasks [{self._id}] launching a new Task:\n"
               f"{self.current_task.description}")
        LOGGER.debug(log)
        self.current_task.start()
        log = (f"Tasks [{self._id}] finished @ {threading.current_thread().name}\n")
        LOGGER.debug(log)
        if self.abort_flag:
            log = (f"任务中止:\n{self.msg.chat_text} @ {self.msg.user_name}\n"
                   f"\t{self.current_task.description}")
            LOGGER.info(log)
        elif self.next_task is not None:
            log = (f"下一项任务:\n{self.msg.chat_text} @ {self.msg.user_name}\n"
                   f"\t{self.next_task.description}")
            LOGGER.info(log)
            TaskQueue.put(self)
        else:
            log = (f"指令已完成:\n{self.msg.chat_text} @ {self.msg.user_name}")
            LOGGER.info(log)

    def reschedule(self, after_t):
        self.next_task.reschedule(after_t)
        log = f'Tasks [{self._id}] has been rescheduled to {self.next_task.start_time}'
        LOGGER.debug(log)

    def __str__(self):
        s = f"Tasks [{self._id}] {self.msg.user_name} @{self.msg.time}: "
        if self.next_task:
            return s + self.next_task.short_description
        else:
            return s

    def _to_dict(self):
        return {
            'msg': self.msg._to_dict(),
            '_task_list': [t._to_dict() for t in self._task_list]
        }

    @classmethod
    def _from_dict(cls, data):
        from ..command import ChatMsg
        from .. import task
        msg = ChatMsg._from_dict(data['msg'])
        ret = cls(msg)
        for d in data['_task_list']:
            cls_ = getattr(task, d['type'], None)
            if cls_ and issubclass(cls_, Task):
                ret._task_list.append(cls_._from_dict(d, ret))
            else:
                raise ValueError(f'Unknown or invalid class type: {d["type"]}')
        return ret
        


class TaskNode:
    def __init__(self,
                 tasks: Tasks,
                 prev: TaskNode | None = None,
                 next: TaskNode | None = None):
        self.tasks = tasks
        if prev is None or next is None:
            self.prev = self
            self.next = self
        else:
            self.prev = prev
            self.next = next

    @staticmethod
    def insert(tasks: Tasks, prev: TaskNode, next: TaskNode):
        new_node = TaskNode(tasks, prev, next)
        new_node.prev.next = new_node
        new_node.next.prev = new_node
    
    @staticmethod
    def remove(node: TaskNode):
        node.prev.next = node.next
        node.next.prev = node.prev


class TaskQueue:
    worker_lock = threading.Lock()
    _instance = None
    _queue_condition = threading.Condition()
    _sentinal_node = TaskNode(None) # type: ignore
    _sentinal_node.prev = _sentinal_node
    _sentinal_node.next = _sentinal_node
    _cur_tasks: Tasks | None = None
    _size = 0
    
    def __new__(cls):
        if not cls._instance:
            with cls._queue_condition:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def empty(cls) -> bool:
        with cls._queue_condition:
            return cls._size == 0

    @classmethod
    def put(cls, tasks: Tasks):
        if not isinstance(tasks, Tasks):
            raise ValueError()
        ut = get_ut()
        if tasks.next_task.start_time < ut:
            tasks.next_task.start_time = ut + 60
        with cls._queue_condition:
            cls._put_helper(tasks)
            cls._size += 1
            log = f"A new Tasks is submitted to TaskQueue:\n{str(tasks)}"
            LOGGER.debug(log)
            cls._queue_condition.notify()

    @classmethod
    def _put_helper(cls, tasks: Tasks):
        if tasks.next_task is None:
            log = f'Cannot submit empty Tasks [{tasks._id}]'
            LOGGER.warning(log)
            return
        newt_st = tasks.next_task.start_time
        newt_et = newt_st + tasks.next_task.duration
        newt_im = tasks.next_task.importance
        conflict_nodes: list[TaskNode] = []
        max_conflict_im = 0

        cur_node = cls._sentinal_node.next
        while cur_node is not cls._sentinal_node:
            if cur_node.tasks.next_task is None:
                log = f'TaskQueue encountered an empty Tasks [{cur_node.tasks._id}] while putting new tasks'
                LOGGER.debug(log)
                cur_node = cur_node.next
                continue
            if cur_node.tasks.next_task.start_time + cur_node.tasks.next_task.duration <= newt_st:
                cur_node = cur_node.next
                continue
            if cur_node.tasks.next_task.start_time >= newt_et:
                break
            conflict_nodes.append(cur_node)
            if cur_node.tasks.next_task.importance > max_conflict_im:
                max_conflict_im = cur_node.tasks.next_task.importance
            cur_node = cur_node.next
        
        if len(conflict_nodes) == 0:
            TaskNode.insert(tasks, cur_node.prev, cur_node)
        elif newt_im < max_conflict_im or newt_im == max_conflict_im < MAX_IMPORTANCE:  # 重新安排新任务
            # TODO: 直接在最后一个冲突的任务之后重排不是最优的
            log = f'Tasks [{tasks._id}] is conflict with other higher importance tasks in TaskQueue, rescheduling after Tasks [{conflict_nodes[-1].tasks._id}]...'
            LOGGER.debug(log)
            after_t = conflict_nodes[-1].tasks.next_task.start_time + conflict_nodes[-1].tasks.next_task.duration # type: ignore
            tasks.reschedule(after_t)
            cls._put_helper(tasks)
        elif max_conflict_im < newt_im:  # 重新安排所有冲突任务
            TaskNode.insert(tasks, conflict_nodes[0].prev, cur_node)
            for n in conflict_nodes:
                log = f'Tasks [{n.tasks._id}] in TaskQueue is conflict with incoming higher importance Tasks [{tasks._id}], rescheduling...'
                LOGGER.debug(log)
                n.tasks.reschedule(newt_et)
                cls._put_helper(n.tasks)
        else:  
            # TODO: 两者均是最高优先级任务, 等待手动控制
            log = f'Two or more highest importance Tasks are conflict, requiring manual rescheduling...'
            LOGGER.debug(log)
            TaskNode.insert(tasks, cur_node.prev, cur_node)

    @classmethod
    def get(cls) -> Tasks:
        with cls._queue_condition:
            while cls._size == 0:
                cls._queue_condition.wait()
            cls._cur_tasks = cls._sentinal_node.next.tasks
            TaskNode.remove(cls._sentinal_node.next)
            cls._size -= 1
            return cls._cur_tasks

    @classmethod
    def clear(cls):
        with cls._queue_condition:
            while cls._size > 0:
                cls.get()

    @classmethod
    def remove_by_user(cls, user_id: str):
        with cls._queue_condition:
            last_node: TaskNode | None = None
            cur_node = cls._sentinal_node.next
            while cur_node is not cls._sentinal_node:
                if cur_node.tasks.msg.user_id == user_id:
                    if last_node is None or cur_node.tasks.msg.time > last_node.tasks.msg.time:
                        last_node = cur_node
                cur_node = cur_node.next
            if last_node is None:
                LOGGER.info(f'删除失败，未找到用户[{user_id}]创建的任务')
                return
            TaskNode.remove(last_node)
            LOGGER.info(f'删除用户[{user_id}]创建的任务:\n{cur_node.tasks}')
            cls._size -= 1
    
    @classmethod
    def remove_by_id(cls, task_id: int):
        with cls._queue_condition:
            cur_node = cls._sentinal_node.next
            while cur_node is not cls._sentinal_node:
                if cur_node.tasks._id == task_id:
                    TaskNode.remove(cur_node)
                    LOGGER.info(f'删除任务:\n{cur_node.tasks}')
                    cls._size -= 1
                    return
                cur_node = cur_node.next
            LOGGER.info(f'删除失败，未找到任务[{task_id}]')

    def __str__(self):
        s = []
        with self._queue_condition:
            if self._cur_tasks and self._cur_tasks.current_task:
                s.append("> 正在执行:")
                s.append(self._cur_tasks.current_task.description)
            if self.empty():
                s.append("> 任务队列为空...\n发送\"!launch\"快速部署一项发射任务")
            else:
                s.append("> 任务队列:")
                cur_node = self._sentinal_node.next
                for _ in range(min(self._size, 5)):
                    s.append(str(cur_node.tasks)) 
                    cur_node = cur_node.next
                if self._size > 5:
                    s.append('...')
            return '\n'.join(s) + '\n'

    @classmethod
    def info(cls):
        return str(cls())

    @classmethod
    def dump_all(cls):
        task_list = []
        while cls._size > 0:
            task_list.append(cls.get()._to_dict())
        return task_list

    @classmethod
    def load_all(cls, data):
        if cls._size > 0:
            LOGGER.warning(f'TaskQueue非空!')
            cls.clear()
        for t in data:
            cls.put(Tasks._from_dict(t))
