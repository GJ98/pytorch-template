import torch
import os
from abc import abstractmethod
import wandb


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = config["trainer"]["epochs"]
        self.mode = config["trainer"]["mode"]
        self.early_stopping = config['trainer']['early_stopping']
        self.patience = config['trainer']['patience'] 
        self.best_score = float("inf") if self.mode == "min" else 0
        self.prev_save_file = None

        # model save folder path
        if not os.path.exists(config["trainer"]["save_dir"]):
            os.makedirs(config["trainer"]["save_dir"])

        # model save file path
        self.save_file = (
            f'{config["trainer"]["save_dir"]}'
            + f'{self.config["run_name"]}'.replace("/", "-")
        )

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):

        for epoch in range(self.epochs):
            self._train_epoch(epoch)

            if self.early_stopping and self.patience < 0:
                return
