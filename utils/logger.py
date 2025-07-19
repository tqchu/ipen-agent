import asyncio

from pentest_server.event import EventType


class QueueLogger:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def log(self, type :EventType, msg: str, hosts: list = None, ports: dict = None, vulnerabilities: dict = None, *args, **kwargs):
        self.queue.put_nowait(
            {
                "type": type.value,
                "message": msg.format(*args, **kwargs),
                "hosts": hosts if hosts else [],
                "ports": ports,
                "vulnerabilities": vulnerabilities,
            }
        )
