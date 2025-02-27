from multiprocessing.connection import Connection

from pytorch_lightning.callbacks import ProgressBar

__all__ = [
    "create_progress_bar",
    "progress_bar_step",
    "remove_progress_bar",
    "ConnProgressBar",
]

_conn = None


def create_progress_bar(conn: Connection, total: int):
    global _conn
    _conn = conn
    _conn.send(total)


def progress_bar_step(n: int):
    global _conn
    if _conn is None:
        return
    _conn.send(n)


def remove_progress_bar():
    global _conn
    if _conn is None:
        return
    _conn.send(0)
    _conn = None


class ConnProgressBar(ProgressBar):
    def __init__(self, conn: Connection, total: int):
        super().__init__()
        create_progress_bar(conn, total)

    def enable(self):
        pass

    def disable(self):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        progress_bar_step(1)
