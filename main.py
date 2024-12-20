import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from time import sleep

from .listener import Listener
from .utils import LOGGER
from .task.tasks import TaskQueue



def listener_loop():
    sleep(5)
    while True:
        command = chat_listener.get()
        if command is None:
            sleep(3)
            continue
        if Listener.is_stop_sign(command):
            asyncio.run(chat_listener._stop())
            sleep(3)
            break
        t = command.process(task_queue)
        if t is None or t.next_task is None:
            continue

        msg = (f'收到新指令:\n{command.msg.chat_text} @ {command.msg.user_name}\n'
               f'{t.next_task.description}')
        LOGGER.info(msg)
        task_queue.put(t)


def worker():
    while True:
        task_queue.worker_lock.acquire()  # 先获取工作锁在获取队列锁避免写者饥饿
        LOGGER.debug(f"Thread [{threading.current_thread().name}] getting new task")
        tasks = task_queue.get()
        if tasks.next_task is None:
            continue

        LOGGER.debug(f'Thread [{threading.current_thread().name}] processing tasks')
        LOGGER.info(f'执行任务:\n{tasks.next_task.description}')
        tasks = tasks.do()

        if tasks is None or tasks.abort_flag == True:
            continue
        task_queue.put(tasks)
        task_queue.worker_lock.release()


def thread_pool():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(3):
            executor.submit(worker)
    print('Thread pool executor finished.')


task_queue = TaskQueue()
worker_lock = threading.Lock()  # 限制只有一个线程可以控制
chat_listener = Listener(27765315, 'e792a714%2C1734606751%2Cb60e3%2A61CjCtESKwChxHgvYUmgEz0oYBgsJ_3a_otvBzV8Agu8NWK8lfbHDj1m4lO_bIwjgwuQQSVkthTzhGYWM1Z0hqYXFxaE9pX1lHQ0RzXzRENThjNnRjdXZLTEQwczlUcUpGNDVwdjA3ZXplcHNBX3BwOGhsaUpNcHFhMXFCdDljYjRWNjJERGlYNGxBIIEC')

chat_thread = threading.Thread(target=listener_loop)
loop_thread = threading.Thread(target=thread_pool)

chat_thread.start()
loop_thread.start()

chat_listener.start()
