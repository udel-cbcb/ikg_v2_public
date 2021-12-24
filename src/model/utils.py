from loguru import logger

class EarlyStopper:

    def __init__(self,stopping_delta,patience):
        self.best_loss = 0
        self.early_stopping_delta = stopping_delta
        self.patience = patience
        self.num_of_epochs_waited = 0
        self.change = 0

    def should_stop(self,loss,epoch_index):
        if epoch_index == 0:
            self.best_loss = loss
            return False
        else:
            # if loss is not better than best loss
            if loss > self.best_loss:

                # how much have we waited for improvement

                # enough time
                if self.num_of_epochs_waited > self.patience:
                    logger.info("Waited more than patience")
                    return True

                # not enough time
                else:
                    self.num_of_epochs_waited = self.num_of_epochs_waited + 1
                    return False     

            # if loss is better than best loss           
            else:

                # how much better
                self.change = self.best_loss - loss

                # not as much as required
                if self.change < self.early_stopping_delta:

                    # how much have we waited for change to improve
                    if self.num_of_epochs_waited > self.patience:
                        logger.info("Change is less early stopping delta and waited more than patience")
                        return True
                    else:
                        self.num_of_epochs_waited = self.num_of_epochs_waited + 1
                        return False

                # is as much as required
                else:
                    self.num_of_epochs_waited = 0
                    self.best_loss = loss
                    return False