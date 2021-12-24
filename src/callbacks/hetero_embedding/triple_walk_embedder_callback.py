from determined.pytorch import PyTorchCallback
from src.model.utils import EarlyStopper
from loguru import logger
import pandas as pd

class TripleWalkCallBack(PyTorchCallback):

    def __init__(self,trial,early_stopping_delta,early_stopping_patience,protein_idx,fold_n,protein_labels,model_type):
        self.trial = trial
        self.early_stopper = EarlyStopper(
            stopping_delta=early_stopping_delta,
            patience=early_stopping_patience
        )
        self.index = 0
        self.protein_idx = protein_idx
        self.fold_n = fold_n
        self.protein_labels = protein_labels
        self.model_type = model_type
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

    def on_checkpoint_end(self, checkpoint_dir: str) -> None:
        rank = self.trial.context.distributed.get_rank()
        if rank == 0:
            # save the embedding
            self.save_embedding(checkpoint_dir)
        else:
            logger.info(f"Skipping save from rank: {rank}")

    def save_embedding(self,checkpoint_dir):
        # save head embeddings
        head_embedding = self.trial.model.get_head_embedding(to_cpu=False)
        out_path = f"{checkpoint_dir}/protein_head_embeddings_{self.fold_n}.csv"
        self._save_embedding(head_embedding,out_path)

        # save tail embeddings
        head_embedding = self.trial.model.get_tail_embedding(to_cpu=False)
        out_path = f"{checkpoint_dir}/protein_tail_embeddings_{self.fold_n}.csv"
        self._save_embedding(head_embedding,out_path)

        logger.info(f"Done")

    def _save_embedding(self,embedding,out_path):

        # get embeddings
        protein_embeddings = embedding[self.protein_idx].cpu().numpy()

        # create embeddings df
        protein_embeddings_df = pd.DataFrame(data=protein_embeddings)
        protein_embeddings_df.insert(0,"protein",self.protein_labels)

        # save
        logger.info(f"Saving embedding to {out_path}")
        protein_embeddings_df.to_csv(out_path,index=False)
        

    