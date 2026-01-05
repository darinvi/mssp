from src.data import DataManager
from src.ols import PairwiseLinearRegression, ensure_x, ensure_y
from src.history import History
import torch
import time

class MSSP:
    def __init__(
        self, 
        test_size=0.2, 
        n_best=300, 
        epochs=20, 
        loss_fn="mape", 
        early_stopping=True, 
        patience=5, 
        random_seed=None,
        allow_diversity=True,
        diversity_ratio=0.25,
        dropna=True,
        pow_cross=True
    ):
        if not isinstance(test_size, float) or test_size <= 0 or test_size >= 1:
            raise ValueError(f"Test size must be a float between 0 and 1, received {test_size} of type {type(test_size)}")
        self.test_size = test_size

        if not isinstance(n_best, int) or n_best <= 0:
            raise ValueError(f"N best must be a positive integer, received {n_best} of type {type(n_best)}")
        self.n_best = n_best
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)

        self.data_manager = DataManager()
        self.history = History()
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.diversity_ratio = diversity_ratio
        self.dropna = dropna
        self.model = PairwiseLinearRegression(metrics=loss_fn)
        self.pow_cross = pow_cross


    @staticmethod
    def _validate_X_y_valid(X_valid, y_valid):
        if X_valid is None and y_valid is None:
            return None, None
        
        if not all(val is not None for val in [X_valid, y_valid]):
            raise ValueError("X_valid and y_valid must be provided together")
        
        if not isinstance(X_valid, torch.Tensor):
            X_valid = torch.as_tensor(X_valid, dtype=torch.float)

        if not isinstance(y_valid, torch.Tensor):
            y_valid = torch.as_tensor(y_valid, dtype=torch.float)

        if X_valid.ndim != 2:
            raise ValueError(f"X_valid must be a 2D tensor, received {X_valid.ndim}D")

        # TODO also tranform X_valid, y_valid

        return X_valid, y_valid


    def _normalize(self, X, X_valid=None):
        if X_valid is None:
            return (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0] + 1e-6) + 1e-6, None
        max = torch.max(X.max(dim=0)[0], X_valid.max(dim=0)[0])
        min = torch.min(X.min(dim=0)[0], X_valid.min(dim=0)[0])
        X = (X - min) / (max - min + 1e-6) + 1e-6
        X_valid = (X_valid - min) / (max - min + 1e-6) + 1e-6
        return X, X_valid


    def _get_index_mask(self, scores):
        indexes = torch.argsort(scores)
        # print(len(scores), torch.isnan(scores).sum())
        # print(indexes)
        # print(scores[indexes])
        mask_best = indexes[:int(self.n_best * (1 - self.diversity_ratio))]
        indexes = indexes[int(self.n_best * (1 - self.diversity_ratio)):]
        mask_rand = indexes[torch.randperm(len(indexes))[:int(self.n_best * self.diversity_ratio)]]
        mask = torch.cat([mask_best, mask_rand])
        return mask

    def _on_epoch_end(self, mask, ep, pow_cross):
        self.history.register({
            'mask': mask,
            'epoch': ep,
            'coef': self.model.coef_,
            'intercept': self.model.intercept_,
            'i': self.model.i[mask],
            'j': self.model.j[mask],
            'type': 'pow_cross_selection' if pow_cross else 'lin_cross_selection'
        })

    def _fit(self, X, y, X_valid, y_valid, ep, pow_cross=False):
        if pow_cross:
            # X, X_valid = self._normalize(X, X_valid)
            X = torch.log(X)
            X_valid = torch.log(X_valid)

        self.model.fit(X, y)
        scores = self.model.evaluate(X_valid, y_valid, pow_cross=pow_cross)
        mask = self._get_index_mask(scores[self.loss_fn])


        X = self.model.predict(X)
        X = X[:, mask]

        X_valid = self.model.predict(X_valid)
        X_valid = X_valid[:, mask]

        if pow_cross:
            # X, X_valid = self._normalize(X, X_valid)
            X = torch.exp(X)
            X_valid = torch.exp(X_valid)

        self._on_epoch_end(mask, ep, pow_cross)

        return X, X_valid, scores[self.loss_fn][mask]


    def _get_best_solutions(self, X, X_valid, scores, epoch):
        mask = self._get_index_mask(scores)
        self.history.register({
            'mask': mask, 
            'epoch': epoch,
            'type': 'best_selection',
        })
        return X[:, mask], X_valid[:, mask]


    @ensure_y
    @ensure_x
    def fit(self, X, y, X_valid=None, y_valid=None):
        # X_valid, y_valid = MSSP._validate_X_y_valid(X_valid, y_valid)
        
        y_cross = torch.log(y) # TODO should have validation for non-negative y. Have some correction in case of negative y. Will then be present in the formula.

        X = self.data_manager.fit(X, y)

        best_loss, patience_counter = float('inf'), 0
        for ep in range(self.epochs):
            st = time.time()
            X_lin, X_valid_lin, scores_lin = self._fit(X, y, X_valid=X, y_valid=y, ep=ep)
            if self.pow_cross:
                X_pow, X_valid_pow, scores_pow = self._fit(X, y_cross, X_valid=X, y_valid=y, ep=ep, pow_cross=True)

            scores = scores_lin if not self.pow_cross else torch.cat([scores_lin, scores_pow])
            X_ep = X_lin if not self.pow_cross else torch.cat([X_lin, X_pow], dim=1)
            X_ep_val = X_valid_lin if not self.pow_cross else torch.cat([X_valid_lin, X_valid_pow], dim=1)
            X, X_valid = self._get_best_solutions(X_ep, X_ep_val, scores, ep)

            # 
            scores = scores[~torch.isnan(scores)]
            # 
            loss_epoch = scores.min()

            print(f'loss ({self.loss_fn}): {loss_epoch.item():.4f}', "epoch:", ep, f", time: {time.time() - st:.2f}s")

            if not self.early_stopping:
                best_loss = min(best_loss, loss_epoch)
                continue

            if loss_epoch < best_loss:
                best_loss = loss_epoch
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