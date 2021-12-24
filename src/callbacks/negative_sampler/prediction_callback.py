from determined.pytorch import PyTorchCallback
from src.model.utils import EarlyStopper
from loguru import logger
import pandas as pd

class PredictionCallBack(PyTorchCallback):

    def __init__(self,trial,early_stopping_delta,early_stopping_patience,fold_n):
        self.trial = trial
        self.fold_n = fold_n
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
            #self.trial.context.set_stop_requested(True)

        # update index
        self.index = self.index + 1

    def on_checkpoint_end(self, checkpoint_dir: str) -> None:
        rank = self.trial.context.distributed.get_rank()
        if rank == 0:
            logger.info("Checkpoint end")
        else:
            logger.info(f"Skipping save from rank: {rank}")
