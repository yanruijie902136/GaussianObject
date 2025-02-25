from pytorch_lightning.callbacks import ProgressBar

__all__ = ["set_progress_bar", "clear_progress_bar", "progress_bar_step", "TkinterProgressBar"]

_wizard = None
_progress_bar = None


def set_progress_bar(wizard, progress_bar, maximum):
    global _wizard, _progress_bar
    _wizard = wizard
    _progress_bar = progress_bar
    _progress_bar.configure(maximum=maximum)


def clear_progress_bar():
    global _wizard, _progress_bar
    _wizard = None
    _progress_bar = None


def progress_bar_step(step=1):
    if _progress_bar is None:
        return
    _progress_bar.step(step)
    _wizard.update()


class TkinterProgressBar(ProgressBar):
    def __init__(self, wizard, progress_bar, max_steps):
        super().__init__()
        self.wizard = wizard
        self.progress_bar = progress_bar
        self.max_steps = max_steps
        set_progress_bar(self.wizard, self.progress_bar, maximum=self.max_steps)
        self.enable = True

    def enable(self):
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        progress_bar_step()
