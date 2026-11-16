"""Training: label-smoothed loss, learning-rate schedules, experiment composition,
checkpointing with exact resumption, and the trainer.

Owner: Training owner. Contract: docs/REPOSITORY.md#srcaiayntraining.
"""

from aiayn.training.config import OptimizerConfig, ScheduleConfig, TrainConfig, load_train_config
from aiayn.training.experiment import Experiment, check_compatibility, load_experiment
from aiayn.training.losses import LossOutput, label_smoothed_nll
from aiayn.training.schedule import LRSchedule, inverse_sqrt_lr, noam_lr
from aiayn.training.trainer import Progress, Trainer, TrainResult, ensure_data

__all__ = [
    "Experiment",
    "LRSchedule",
    "LossOutput",
    "OptimizerConfig",
    "Progress",
    "ScheduleConfig",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "check_compatibility",
    "ensure_data",
    "inverse_sqrt_lr",
    "label_smoothed_nll",
    "load_experiment",
    "load_train_config",
    "noam_lr",
]
