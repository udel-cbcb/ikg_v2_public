from determined.pytorch import PyTorchCallback
from src.model.utils import EarlyStopper
from loguru import logger
import pandas as pd

class DeepWalkCallBack(PyTorchCallback):

    def __init__(self,trial,early_stopping_delta,early_stopping_patience):
        self.trial = trial
        self.early_stopper = EarlyStopper(
            stopping_delta=early_stopping_delta,
            patience=early_stopping_patience
        )
        self.index = 0
        super().__init__()

    def on_validation_end(self, metrics) -> None:
       
        # check if need to early stop
        val_loss = metrics["val_loss"]
        should_stop = self.early_stopper.should_stop(val_loss,self.index)
        logger.info(f"Index: {self.index} / Best loss: {self.early_stopper.best_loss} / Change: {self.early_stopper.change} / Waited: {self.early_stopper.num_of_epochs_waited}")

        if should_stop:
            logger.info("Requesting early stopping")

            # request stop
            self.trial.context.set_stop_requested(True)

        # update index
        self.index = self.index + 1

    