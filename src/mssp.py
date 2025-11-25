from src.data import DataManager
from src.ols import PairwiseLinearRegression, ensure_x, ensure_y
from src.history import History
import torch

class MSSP:
    def __init__(
        self, 
        test_size=0.2, 
        n_best=100, 
        epochs=300, 
        loss_fn="mape", 
        early_stopping=True, 
        patience=5, 
        add_powers=False, 
        add_synergies=False,
        random_seed=None,
        allow_diversity=True,
        diversity_ratio=0.25
    ):
        if not isinstance(test_size, float) or test_size <= 0 or test_size >= 1:
            raise ValueError(f"Test size must be a float between 0 and 1, received {test_size} of type {type(test_size)}")
        self.test_size = test_size

        if not isinstance(n_best, int) or n_best <= 0:
            raise ValueError(f"N best must be a positive integer, received {n_best} of type {type(n_best)}")
        self.n_best = n_best
        
        self.data_manager = DataManager(add_powers=add_powers, add_synergies=add_synergies)
        self.model = PairwiseLinearRegression()
        self.history = History()
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_seed = random_seed
        self.allow_diversity = allow_diversity
        self.diversity_ratio = diversity_ratio

    def _init_train_test_indexes(self, n):
        indexes = torch.randperm(n)
        self.i_tr = indexes[:int(n * (1 - self.test_size))]
        self.i_te = indexes[int(n * (1 - self.test_size)):]


    @ensure_y
    @ensure_x
    def fit(self, X, y):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)

        # self._init_train_test_indexes(X.shape[0])

        # X = self.data_manager.fit(X[self.i_tr], y[self.i_tr])
        
        # TODO use tr te splits
        X = self.data_manager.fit(X, y)
        # 

        # TODO check primitives; avoid NaNs
        X = X[:, ~X.isnan().all(dim=0)]

        best_loss = float('inf')
        patience_counter = 0
        for ep in range(self.epochs):
            # self.model.fit(X[self.i_tr], y[self.i_tr])
            # scores = self.model.evaluate(X[self.i_te], y[self.i_te], metrics=[self.loss_fn])

            # TODO use tr te splits
            self.model.fit(X, y)
            scores = self.model.evaluate(X, y, metrics=[self.loss_fn])
            # 

            # Get best 75% and 25% randomly for diversity
            indexes = torch.argsort(scores[self.loss_fn])
            if self.allow_diversity:
                mask_best = indexes[:int(self.n_best * (1 - self.diversity_ratio))]
                indexes = indexes[int(self.n_best * (1 - self.diversity_ratio)):]
                mask_rand = indexes[torch.randperm(len(indexes))[:int(self.n_best * self.diversity_ratio)]]
                mask = torch.cat([mask_best, mask_rand])
            else:
                mask = indexes[:self.n_best]

            # history.register()
            X = self.model.predict(X)
            X = X[:, mask]

            loss = scores[self.loss_fn].min()
            print(f'loss ({self.loss_fn}): {loss.item():.4f}', "epoch:", ep)

            if self.early_stopping and loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        print(f"Best loss: {best_loss} after training for {ep} epochs")

    @ensure_x
    def predict(self, X):
        X = self.data_manager.transform(X)
        return self.model.predict(X)