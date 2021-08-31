class EarlyStopping:
    def __init__(self, epochs_wait):
        self.num_bad_epochs = 0
        self.bad_epochs_max = epochs_wait
        self.best_loss = None

    def step(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    def is_stop(self):
        if self.num_bad_epochs == self.bad_epochs_max:
            print("Early Stopping.")
            return True
        else:
            return False