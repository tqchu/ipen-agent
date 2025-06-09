import asyncio

from pentest_server.event import EventType


class QueueLogger:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def log(self, type :EventType, msg: str, *args, **kwargs):
        self.queue.put_nowait(
            {
                "type": type.value,
                "message": msg.format(*args, **kwargs)
            }
        )
