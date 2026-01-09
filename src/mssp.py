from src.data import DataManager
from src.ols import PairwiseLinearRegression, ensure_x, ensure_y
from src.tree import ModelTree
import torch
import time

class MSSP:
    def __init__(
        self, 
        n_best=100, 
        epochs=10, 
        loss_fn="mape", 
        early_stopping=True, 
        patience=5, 
        random_seed=None,
        allow_diversity=True,
        diversity_ratio=0.75,
        dropna=True,
        pow_cross=True
    ):
        if not isinstance(n_best, int) or n_best <= 0:
            raise ValueError(f"N best must be a positive integer, received {n_best} of type {type(n_best)}")
        self.n_best = n_best
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)

        self.loss_fn = loss_fn
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.diversity_ratio = diversity_ratio
        self.dropna = dropna
        self.pow_cross = pow_cross
        self.built = False

    def _init_objects(self):
        self.data_manager = DataManager()
        self.ols = PairwiseLinearRegression(metrics=self.loss_fn)
        self.history = []
        self.built = False
        self.model = []


    def register(self, data):
        self.history.append(data)


    def _normalize(self, X, X_valid=None):
        if X_valid is None:
            return (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0] + 1e-6) + 1e-6, None
        max = torch.max(X.max(dim=0)[0], X_valid.max(dim=0)[0])
        min = torch.min(X.min(dim=0)[0], X_valid.min(dim=0)[0])
        X = (X - min) / (max - min + 1e-6) + 1e-6
        X_valid = (X_valid - min) / (max - min + 1e-6) + 1e-6
        return X, X_valid


    def _on_epoch_end(self, mask, ep, pow_cross):
        msg = {
            'mask': mask.clone().cpu(),
            'epoch': ep,
            'coef': self.ols.coef_[mask, :].clone().cpu(),
            'intercept': self.ols.intercept_[mask].clone().cpu(),
            'i': self.ols.i[mask].clone().cpu(),
            'j': self.ols.j[mask].clone().cpu(),
            'type': 'pow_cross_selection' if pow_cross else 'lin_cross_selection',
        }

        self.register(msg)


    def _fit(self, X, y, X_valid, y_valid, ep, pow_cross=False):
        if pow_cross:
            X = torch.log(X)
            if X_valid is not None:
                X_valid = torch.log(X_valid)

        self.ols.fit(X, y)
        
        if X_valid is not None: 
            scores = self.ols.evaluate(X_valid, y_valid, pow_cross=pow_cross)
        else:
            scores = self.ols.evaluate(X, y, pow_cross=pow_cross)

        mask = self._get_index_mask(scores[self.loss_fn])

        X = self.ols.predict(X)
        X = X[:, mask]

        if X_valid is not None:
            X_valid = self.ols.predict(X_valid)
            X_valid = X_valid[:, mask]

        if pow_cross:
            X = torch.exp(X)
            if X_valid is not None:
                X_valid = torch.exp(X_valid)

        self._on_epoch_end(mask, ep, pow_cross)

        return X, X_valid, scores[self.loss_fn][mask]


    def _get_best_solutions(self, X, X_valid, scores, epoch):
        mask = self._get_index_mask(scores.clone().cpu())
        self.register({
            'mask': mask, 
            'epoch': epoch,
            'type': 'best_selection',
        })
        if X_valid is not None:
            X_valid = X_valid[:, mask]
        return X[:, mask], X_valid


    def _get_index_mask(self, scores):
        '''
        Get ranks of scores low to high and sort scores.
        '''
        indexes = torch.argsort(scores)
        scores = scores[indexes]

        '''
        Check whether there are bad scores (log can introduse).
        This is important as if not excluded, the random mask might choose them.
        '''
        bad_mask = ~torch.isfinite(scores)
        if bad_mask.any():
            i = bad_mask.nonzero()[0, 0]
            indexes = indexes[:i]

        '''
        Mask the n_best for next generation. Introduce some randomness.
        '''
        mask = indexes[:int(self.n_best * (1 - self.diversity_ratio))]
        indexes = indexes[int(self.n_best * (1 - self.diversity_ratio)):]
        if indexes.shape[0] != 0:
            mask_rand = indexes[torch.randperm(len(indexes))[:int(self.n_best * self.diversity_ratio)]]
            mask = torch.cat([mask, mask_rand])
        return mask


    @ensure_y
    @ensure_x
    def fit(self, X, y, X_valid=None, y_valid=None):
        '''
        Init inside fit so a subsequent call to the method resets the state and history is correct.
        '''
        self._init_objects()

        X, params = self.data_manager.fit(X, y)

        X_valid, y_valid = self._validate_X_y_valid(X_valid, y_valid)
        
        if self.pow_cross:
            if y.min() <= 0:
                raise Exception("Negative values in the target, handle appropriate for pow cross method.")
            y_cross = torch.log(y)
            y_valid_cross = torch.log(y_valid) if y_valid is not None else y_cross

        self.register({
            'type': 'primitives',
            'epoch': -1,
            'params': params
        })

        best_loss, patience_counter = float('inf'), 0
        for ep in range(self.epochs):
            st = time.time()

            '''
            Fitting all pairwise and getting the n_best + a portion of random candidates for each method (lin, pow).
            '''
            X_lin, X_valid_lin, scores_lin = self._fit(X, y, X_valid=X_valid, y_valid=y_valid, ep=ep)
            if self.pow_cross:
                X_pow, X_valid_pow, scores_pow = self._fit(X, y_cross, X_valid=X_valid, y_valid=y_valid_cross, ep=ep, pow_cross=True)


            '''
            Getting n_best + a portion of random candidates from the overall population.
            '''
            scores, X, X_valid = scores_lin, X_lin, X_valid_lin
            if self.pow_cross:
                scores = torch.cat([scores_lin, scores_pow])
                X = torch.cat([X_lin, X_pow], dim=1)
                if X_valid is not None:
                    X_valid = torch.cat([X_valid_lin, X_valid_pow], dim=1)
                    
            X, X_valid = self._get_best_solutions(X, X_valid, scores, ep)


            '''
            Early stopping logic.
            '''
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


    def _validate_X_y_valid(self, X_valid, y_valid):
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

        X_valid = self.data_manager.transform(X_valid)

        return X_valid, y_valid


    @ensure_x
    def predict(self, X, top_k=1, clip=None):
        if clip is None:
            clip = self.pow_cross
        self._build_model(top_k)
        X = self.data_manager.transform(X, apply_primitives=False, clip=clip)
        return self.model[0].predict(X).flatten()

    def _build_model(self, top_k):
        '''
        Validations
        '''
        if not isinstance(top_k, int):
            raise TypeError(f"top_k should be int, got {type(top_k)}")

        if top_k < 0 or top_k > self.n_best:
            raise ValueError(f"top_k should be an int between 1 and n_best ({self.n_best})")

        '''
        Required number already built, no need to rebuild models.
        '''
        if (lm := len(self.model)) == top_k:
            return

        '''
        If more models than required, cut down to top_k.
        '''
        if lm > top_k:
            self.model = self.model[:top_k]
            return
        
        '''
        If there are already multiple models built, only build required to get up to top_k, no need to rebuild models.
        '''

        self.model.extend([ModelTree(self.history, i) for i in range(lm, top_k)])