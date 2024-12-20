import asyncio
import http.cookies
import queue
import aiohttp

from .blivedm import BLiveClient, BaseHandler
from .blivedm.models import web as web_models
from .command import Command, ShortCommand, ChatMsg
from .utils import LOGGER


ROOT = "650021430"


class Listener:
    def __init__(self, room_id: int, sessdata: str):
        self.room_id= room_id
        self.sessdata= sessdata
        self._msg_queue: queue.Queue[web_models.DanmakuMessage] = queue.Queue()
        self._incomplete_chat_dic = {}
        self._session: aiohttp.ClientSession | None = None
        self._client: BLiveClient | None = None
        self._started = False

    def start(self):
        if self._started:
            LOGGER.warning('Listener [%s] has already started', self.room_id)
            return
        self._started = True
        asyncio.run(self._listen_task())

    async def _stop(self):
        if self._client is None or not self._started:
            return
        await self._client.stop_and_close()
        self._started = False
        LOGGER.info('Listener [%s] stopped successfully', self.room_id)
        
    def get(self, timeout=None) -> Command | None:
        if not self._started:
            LOGGER.warning('Listener [%s] has not started yet', self.room_id)
            return
        message = self._msg_queue.get(timeout=timeout)
        command = self._chat_processor(message)
        if command is None:
            return
        # 收到终止指令，向其他线程传递指令
        if self.is_stop_sign(command):
            self._msg_queue.put(message)
        self._msg_queue.task_done()
        return command

    @staticmethod
    def is_stop_sign(command: Command):
        return command.msg.user_id == str(ROOT) and command.msg.chat_text == '!stop'

    async def _listen_task(self):
        self._init_session()
        try:
            await self._run_single_client()
        finally:
            await self.session.close()

    def _init_session(self):
        cookies = http.cookies.SimpleCookie()
        cookies['SESSDATA'] = self.sessdata
        cookies['SESSDATA']['domain'] = 'bilibili.com'

        self.session = aiohttp.ClientSession()
        self.session.cookie_jar.update_cookies(cookies)

    async def _run_single_client(self):
        self._client = BLiveClient(self.room_id, session=self.session)
        handler = ListenerHandler(self._msg_queue)
        self._client.set_handler(handler)

        self._client.start()
        try:
            await self._client.join()
        finally:
            await self._client.stop_and_close()

    def _chat_processor(self, message: web_models.DanmakuMessage):
        chat_msg = ChatMsg(str(message.rnd), message.msg, str(message.uid), message.uname, message.timestamp)
        text = chat_msg.chat_text
        nickname = chat_msg.user_name
        if text[0] == '!' and text[-1] == '/':  # 不完整指令的第一句
            self._incomplete_chat_dic[nickname] = text[:-1] + ' '
        elif text[0] == '/' and text[-1] == '/' and nickname in self._incomplete_chat_dic.keys():  # 不完整指令的中间句
            self._incomplete_chat_dic[nickname] += text[1:-1] + ' '
        elif text[0] == '/' and nickname in self._incomplete_chat_dic.keys():  # 不完整指令的最后一句
            self._incomplete_chat_dic[nickname] += text[1:]
            chat_msg.chat_text = self._incomplete_chat_dic[nickname]
            self._incomplete_chat_dic.pop(nickname, None)
            return Command(chat_msg)
        elif text[0] == '!':  # 短指令
            return ShortCommand(chat_msg)


class ListenerHandler(BaseHandler):
    def __init__(self, msg_queue: queue.Queue[web_models.DanmakuMessage]):
        super().__init__()
        self._msg_queue = msg_queue

    def _on_danmaku(self, client: BLiveClient, message: web_models.DanmakuMessage):
        LOGGER.info('%s: %s', message.uname, message.msg)
        self._msg_queue.put(message)

    def _on_gift(self, client: BLiveClient, message: web_models.GiftMessage):
        print(f'{message.uname} 赠送{message.gift_name}x{message.num}'
              f' （{message.coin_type}瓜子x{message.total_coin}）')

    def _on_super_chat(self, client: BLiveClient, message: web_models.SuperChatMessage):
        print(f'醒目留言 ¥{message.price} {message.uname}：{message.message}')


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep

    listener = Listener(27765315, 'e792a714%2C1734606751%2Cb60e3%2A61CjCtESKwChxHgvYUmgEz0oYBgsJ_3a_otvBzV8Agu8NWK8lfbHDj1m4lO_bIwjgwuQQSVkthTzhGYWM1Z0hqYXFxaE9pX1lHQ0RzXzRENThjNnRjdXZLTEQwczlUcUpGNDVwdjA3ZXplcHNBX3BwOGhsaUpNcHFhMXFCdDljYjRWNjJERGlYNGxBIIEC')

    with ThreadPoolExecutor(max_workers=3) as executor:
        def thread_task():
            sleep(5)
            while True:
                command = listener.get()
                if command is None:
                    sleep(3)
                    continue
                if Listener.is_stop_sign(command):
                    asyncio.run(listener._stop())
                    sleep(3)
                    break

        for _ in range(3):  
            executor.submit(thread_task)

        listener.start()
