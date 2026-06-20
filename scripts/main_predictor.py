"""Main training client for standalone predictor."""
import os

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from pedgen.dataset.datamodule import PedGenDataModule
from pedgen.model.predictor_model import PredictorModel

OmegaConf.register_new_resolver("eval", eval)


class LoggerSaveConfigCallback(SaveConfigCallback):
    """Save config callback to log config to the logger."""

    def save_config(self, trainer: Trainer, pl_module: LightningModule,
                    stage: str) -> None:
        if trainer.logger is not None and hasattr(trainer.logger, "log_hyperparams"):
            trainer.logger.log_hyperparams(self.config)


class MyLightningCLI(LightningCLI):
    """Custom LightningCLI with additional arguments."""

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--exp_root',
                            type=str,
                            default="experiments",
                            help='root of experiments')
        parser.add_argument('--exp_name',
                            type=str,
                            default="predictor",
                            help='experiment name')
        parser.add_argument('--version',
                            type=str,
                            default="with_context",
                            help='experiment version')
        parser.add_argument('--force_overwrite', action="store_true")

    @rank_zero_only
    def before_instantiate_classes(self):
        """Create log dir before init."""
        torch.set_float32_matmul_precision('high')
        sub_config = self.config[self.config["subcommand"]]
        log_dir = os.path.join(sub_config["exp_root"], sub_config["exp_name"],
                               sub_config["version"])
        sub_config["trainer"]["default_root_dir"] = log_dir

        if "logger" in sub_config["trainer"] and sub_config["trainer"]["logger"]:
            sub_config["trainer"]["logger"][0]["init_args"]["save_dir"] = log_dir
            sub_config["trainer"]["logger"][0]["init_args"][
                "name"] = f"{sub_config['exp_name']}_{sub_config['version']}"

        sub_config["trainer"]["callbacks"][0]["init_args"][
            "dirpath"] = os.path.join(log_dir, "ckpts")
        os.makedirs(log_dir, exist_ok=True)


def cli_main():
    """Main Function."""
    MyLightningCLI(PredictorModel,
                   PedGenDataModule,
                   save_config_callback=LoggerSaveConfigCallback,
                   save_config_kwargs={"overwrite": True},
                   parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == '__main__':
    cli_main()