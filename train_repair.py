import argparse
import logging
import os
import sys

from progress_bar import *


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


class TrainRepair:
    def __init__(self, wizard=None, progress_bar=None):
        self.wizard = wizard
        self.progress_bar = progress_bar

    def main(self, source_path=None, sparse_num=None):
        parser = self.create_parser()

        if source_path is None:
            args, extras = parser.parse_known_args(sys.argv[1:])
        else:
            args, extras = parser.parse_known_args(self.create_argv(source_path, sparse_num))

        # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
        selected_gpus = [0]

        # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
        # As far as Pytorch Lightning is concerned, we always use all available GPUs
        # (possibly filtered by CUDA_VISIBLE_DEVICES).
        # devices = -1
        if len(env_gpus) > 0:
            # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
            n_gpus = len(env_gpus)
        else:
            selected_gpus = list(args.gpu.split(","))
            n_gpus = len(selected_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        import pytorch_lightning as pl
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
        from lightning_utilities.core.rank_zero import rank_zero_only

        if args.typecheck:
            from jaxtyping import install_import_hook

            install_import_hook("threestudio", "typeguard.typechecked")

        import threestudio
        from threestudio.systems.base import BaseSystem
        from threestudio.utils.callbacks import (
            ConfigSnapshotCallback,
            CustomProgressBar,
        )
        from threestudio.utils.config import ExperimentConfig, load_config
        from threestudio.utils.misc import get_rank

        logger = logging.getLogger("pytorch_lightning")
        if args.verbose:
            logger.setLevel(logging.DEBUG)

        for handler in logger.handlers:
            if handler.stream == sys.stderr:  # type: ignore
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())

        # parse YAML config to OmegaConf
        cfg: ExperimentConfig
        cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

        # set a different seed for each device
        pl.seed_everything(cfg.seed + get_rank(), workers=True)

        # pre load dataset for scene info
        dm = threestudio.find(cfg.data_type)(cfg.data)
        gt_ds = threestudio.find(cfg.dataset_type)(cfg.data, sparse_num=cfg.data.sparse_num)
        cfg.system.scene_extent = gt_ds.get_scene_extent()['radius']  # type: ignore

        system: BaseSystem = threestudio.find(cfg.system_type)(
            cfg.system, resumed=cfg.resume is not None
        )
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

        callbacks = []
        if args.train:
            callbacks += [
                ModelCheckpoint(
                    dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
                ),
                LearningRateMonitor(logging_interval="step"),
                ConfigSnapshotCallback(
                    args.config,
                    cfg,
                    os.path.join(cfg.trial_dir, "configs"),
                    use_version=False,
                ),
                # CustomProgressBar(refresh_rate=1),
            ]
            if self.wizard is not None:
                callbacks.append(TkinterProgressBar(self.wizard, self.progress_bar, cfg.trainer["max_steps"]))

        def write_to_text(file, lines):
            with open(file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

        loggers = []
        if args.train:
            # make tensorboard logging dir to suppress warning
            rank_zero_only(
                lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
            )()
            loggers += [
                TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
                CSVLogger(cfg.trial_dir, name="csv_logs"),
            ]
            rank_zero_only(
                lambda: write_to_text(
                    os.path.join(cfg.trial_dir, "cmd.txt"),
                    ["python " + " ".join(sys.argv), str(args)],
                )
            )()

        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            inference_mode=False,
            accelerator="gpu",
            devices=1,
            **cfg.trainer,
        )

        print("===== fit model =====")
        input()

        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)

        clear_progress_bar()

    def create_argv(self, source_path, sparse_num):
        dataset_name = os.path.basename(source_path)
        return [
            "--config", "configs/gaussian-object-colmap-free.yaml",
            "--train",
            "--gpu", "0",
            f'tag="{dataset_name}"',
            f'system.init_dreamer="output/gs_init/{dataset_name}"',
            f'system.exp_name="output/controlnet_finetune/{dataset_name}"',
            "system.refresh_size=8",
            f'data.data_dir="{source_path}"',
            "data.resolution=8",
            f"data.sparse_num={sparse_num}",
            'data.prompt="a photo of a xxy5syt00"',
            f'data.json_path="output/gs_init/{dataset_name}/refined_cams.json"',
            "data.refresh_size=8",
            "system.sh_degree=2",
        ]

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="path to config file")
        parser.add_argument(
            "--gpu",
            default="0",
            help="GPU(s) to be used. 0 means use the 1st available GPU. "
            "1,2 means use the 2nd and 3rd available GPU. "
            "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
            "this argument is ignored and all available GPUs are always used.",
        )

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--train", action="store_true")
        group.add_argument("--validate", action="store_true")
        group.add_argument("--test", action="store_true")
        group.add_argument("--export", action="store_true")

        parser.add_argument(
            "--verbose", action="store_true", help="if true, set logging level to DEBUG"
        )

        parser.add_argument(
            "--typecheck",
            action="store_true",
            help="whether to enable dynamic type checking",
        )

        return parser


if __name__ == "__main__":
    TrainRepair().main()
