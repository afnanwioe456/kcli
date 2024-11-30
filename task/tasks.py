from __future__ import annotations
import krpc
import threading
from typing import TYPE_CHECKING

from krpc.client import Client
from krpc.services.spacecenter import Vessel
from krpc.event import Event as krpcEvent
from ..utils import LOGGER, UTIL_CONN, logging_around

if TYPE_CHECKING:
    from command import ChatMsg

MAX_IMPORTANCE = 9
"""
IMPORTANCE:
0: 弹幕普通发射任务 
1: 弹幕高优先级发射任务
2: 弹幕紧急发射任务
3: 对接
4: 
5: 默认优先级
6:
7: 默认轨道机动
8: 交会机动
9: 捕获机动
"""


class Task:
    conn: Client | None
    event_list: list[krpcEvent]

    def __init__(self,
                 name: str,
                 tasks: Tasks,
                 start_time: int,
                 duration: int,
                 importance: int = 3,
                 ):
        self.name = name
        self.tasks = tasks
        self.importance = importance
        self.start_time = start_time
        self.duration = duration

        self.event_list = []
        self.time_out = 1800

    @property
    def description(self) -> str:
        return f"undefined task"

    @property
    def short_description(self) -> str:
        return self.description.split('\n')[0]

    def reschedule(self, after_t) -> Task:
        self.start_time = after_t
        return self

    def __str__(self):
        return self.description

    def _conn_setup(self, conn_name: str):
        self.conn = krpc.connect(conn_name)
        self.sc = self.conn.space_center
        if self.sc is None:
            return False
        self.sc.quicksave()
        self.vessel = self.sc.active_vessel
        self.mj = self.conn.mech_jeb # type: ignore
        self.stage = self.vessel.control.current_stage
        if self.conn.krpc is None:
            return False
        self.expression = self.conn.krpc.Expression
        return True

    def _activate_next_stage(self) -> list[Vessel]:
        jettisoned_vessel_list = self.vessel.control.activate_next_stage()
        self.stage = self.vessel.control.current_stage
        print(f"{self.vessel.name}: S{self.stage}")
        return jettisoned_vessel_list

    def _act_part_list_by_type(self,
                               target: str = "engine",
                               stage: int | None = None) -> list:
        if stage is None:
            stage = self.stage
        parts_in_stage = self.vessel.parts.in_stage(stage)
        part_list = []
        for i in parts_in_stage:
            if getattr(i, target):
                part_list.append(getattr(i, target))
        return part_list

    def _sep_part_list_by_name(self,
                               name: str,
                               stage: int | None = None) -> list:
        if stage is None:
            stage = self.stage
        parts_in_stage = self.vessel.parts.in_decouple_stage(stage)
        part_list = []
        for i in parts_in_stage:
            if name in i.title:
                part_list.append(i)
        return part_list

    def _wait_for_events(self):
        while self.event_list:
            event = self.event_list.pop()
            with event.condition:
                try:
                    event.wait(timeout=self.time_out)
                except RuntimeError as e:
                    print("EventRuntimeError:", e)
                    break
            event.remove()
            break
            # TODO: 并发等待需要完善
        if self.event_list:
            [event.remove() for event in self.event_list]
            self.event_list.clear()

    @logging_around
    def start(self):
        self._conn_setup('undefined')


class Tasks:
    current_task: Task | None
    task_list: list[Task]

    def __init__(self,
                 msg: ChatMsg,
                 id: int,
                 task_queue: TaskQueue,
                 ):
        """
        任务序列，一个指令或某个任务自身产生的一系列任务
        """
        self.current_task: Task | None = None
        self.task_list = []  # 等待执行的任务序列
        self.msg = msg
        self.id = id  # 对应的指令编号
        self.task_queue = task_queue
        self.abort_flag = False  # 某一个任务异常终止则放弃整个
    
    @property
    def next_task(self) -> Task | None:
        return self.task_list[0] if self.task_list else None

    def submit(self, task: Task | list[Task]):
        if not isinstance(task, list):
            task = [task]
        self.task_list += task

        log = (f"Tasks [{self.id}] {self.msg.user_name} added {len(task)} new Task(s):\n"
               f"{task[0].short_description}")
        LOGGER.debug(log)

    def submit_nowait(self, task: Task | list[Task]):
        if not isinstance(task, list):
            task = [task]
        self.task_list = task + self.task_list

        log = (f"Tasks [{self.id}] {self.msg.user_name} added {len(task)} new Task(s):\n"
               f"{task[0].short_description}")
        LOGGER.debug(log)

    def do(self) -> Tasks | None:
        if not self.next_task:
            return
        self.current_task = self.task_list.pop(0)
        self.task_queue.write()
        log = (f"Tasks [{self.id}] launching a new Task:\n"
               f"{self.current_task.description}")
        LOGGER.debug(log)

        self.current_task.start()

        log = (f"Tasks [{self.id}] finished @ {threading.current_thread().name}\n")
        LOGGER.debug(log)
        ret: Tasks | None = None
        if self.abort_flag:
            log = (f"任务中止:\n{self.msg.chat_text} @ {self.msg.user_name}\n"
                   f"\t{self.current_task.description}")
            LOGGER.info(log)
        elif self.next_task is not None:
            log = (f"下一项任务:\n{self.msg.chat_text} @ {self.msg.user_name}\n"
                   f"\t{self.next_task.description}")
            LOGGER.info(log)
            ret = self
        else:
            log = (f"指令已完成:\n{self.msg.chat_text} @ {self.msg.user_name}")
            LOGGER.info(log)

        return ret

    def reschedule(self, after_t):
        self.next_task = self.next_task.reschedule(after_t)
        log = f'Tasks [{self.id}] has been rescheduled to {self.next_task.start_time}'
        LOGGER.debug(log)

    def __str__(self):
        s = f"Tasks [{self.id}] {self.msg.user_name} @{self.msg.time}: "
        if self.next_task:
            return s + self.next_task.short_description
        else:
            return s + "N/A"


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
    _queue_condition = threading.Condition()
    # 被保护数据
    _sentinal_node = TaskNode(None) # type: ignore
    _sentinal_node.prev = _sentinal_node
    _sentinal_node.next = _sentinal_node
    _cur_tasks: Tasks | None = None
    _size = 0

    def empty(self) -> bool:
        with self._queue_condition:
            return self._size == 0

    def put(self, tasks: Tasks):
        ut = get_ut()
        if tasks.next_task.start_time < ut:
            tasks.next_task.start_time = ut + 60
        with self._queue_condition:
            self._put_helper(tasks)
            self._size += 1
            log = f"A new Tasks is submitted to TaskQueue:\n{str(tasks)}"
            LOGGER.debug(log)
            self._queue_condition.notify()

    def _put_helper(self, tasks: Tasks):
        if tasks.next_task is None:
            log = f'Cannot submit empty Tasks [{tasks.id}]'
            LOGGER.warning(log)
            return
        newt_st = tasks.next_task.start_time
        newt_et = newt_st + tasks.next_task.duration
        newt_im = tasks.next_task.importance
        conflict_nodes: list[TaskNode] = []
        max_conflict_im = 0

        cur_node = self._sentinal_node.next
        while cur_node != self._sentinal_node:
            if cur_node.tasks.next_task is None:
                log = f'TaskQueue encountered an empty Tasks [{cur_node.tasks.id}] while putting new tasks'
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
            log = f'Tasks [{tasks.id}] is conflict with other higher importance tasks in TaskQueue, reschedualing after Tasks [{conflict_nodes[-1].tasks.id}]...'
            LOGGER.debug(log)
            after_t = conflict_nodes[-1].tasks.next_task.start_time + conflict_nodes[-1].tasks.next_task.duration # type: ignore
            tasks.reschedule(after_t)
            self._put_helper(tasks)
        elif max_conflict_im < newt_im:  # 重新安排所有冲突任务
            TaskNode.insert(tasks, conflict_nodes[0].prev, cur_node)
            for n in conflict_nodes:
                log = f'Tasks [{n.tasks.id}] in TaskQueue is conflict with incoming higher importance Tasks [{tasks.id}], reschedualing...'
                LOGGER.debug(log)
                n.tasks.reschedule(newt_et)
                self._put_helper(n.tasks)
        else:  
            # TODO: 两者均是最高优先级任务, 等待手动控制
            log = f'Two or more highest importance Tasks are conflict, requiring manual reschedualing...'
            LOGGER.debug(log)
            TaskNode.insert(tasks, cur_node.prev, cur_node)

    def get(self) -> Tasks:
        with self._queue_condition:
            while self._size == 0:
                self._queue_condition.wait()
            self._cur_tasks = self._sentinal_node.next.tasks
            TaskNode.remove(self._sentinal_node.next)
            return self._cur_tasks

    def remove_by_user(self, user_id: str):
        """
        删除等待队列以及任务队列中user创建的最新的tasks
        """
        with self._queue_condition:
            last_node: TaskNode | None = None
            cur_node = self._sentinal_node.next
            while cur_node != self._sentinal_node:
                if cur_node.tasks.msg.user_name == user_id:
                    if last_node is None or cur_node.tasks.msg.time > last_node.tasks.msg.time:
                        last_node = cur_node
                cur_node = cur_node.next
            if last_node is None:
                log = f'删除失败，未找到用户[{user_id}]创建的任务'
                LOGGER.info(log)
                return
            TaskNode.remove(last_node)
            log = f'删除用户[{user_id}]创建的任务:\n{cur_node.tasks}'
            self._size -= 1
    
    def remove_by_id(self, task_id: str):
        with self._queue_condition:
            cur_node = self._sentinal_node.next
            while cur_node != self._sentinal_node:
                if cur_node.tasks.id == task_id:
                    log = f'删除任务:\n{cur_node.tasks}'
                    LOGGER.info(log)
                    self._size -= 1
                    break
                cur_node = cur_node.next
            log = f'删除失败，未找到任务[{task_id}]'
            LOGGER.info(task_id)

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

    def write(self):
        """
        向文件写入task_queue
        """
        pass

    def save(self):
        pass


if __name__ == '__main__':
    from command import ChatMsg

    def get_ut():
        return 100

    msg1 = ChatMsg("1", "1", "1", "user1", 1)
    msg2 = ChatMsg("2", "2", "2", "user2", 2)
    msg3 = ChatMsg("3", "3", "1", "user1", 3)

    TQ = TaskQueue()

    tasks1 = Tasks(msg1, 1, TQ)
    tasks2 = Tasks(msg2, 2, TQ)
    tasks3 = Tasks(msg3, 3, TQ)

    task1 = Task("task1", tasks1, 0, 50)
    tasks1.submit(task1)
    task2 = Task("task2", tasks2, 70, 70, importance=0)
    tasks2.submit(task2)
    task3 = Task("task3", tasks3, 0, 30, importance=7)
    tasks3.submit(task3)

    TQ.put(tasks1)
    print("TQ:\n" + str(TQ))
    TQ.put(tasks2)
    print("TQ:\n" + str(TQ))
    TQ.put(tasks3)
    print("TQ:\n" + str(TQ))

    t = TQ.get()
    print(t.next_task.start_time) # type: ignore
    t = TQ.get()
    print(t.next_task.start_time) # type: ignore
    t = TQ.get()
    print(t.next_task.start_time) # type: ignore









