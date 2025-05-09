import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EEGClassifierPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, csp, spectrogram, classifier, csp_components=None,
                 f_range=(8, 30), t_range=(0, 5)):
        self.csp = csp
        self.spectrogram = spectrogram
        self.classifier = classifier
        self.csp_components = csp_components
        self.f_range = f_range
        self.t_range = t_range
        self.is_fitted = False

    def fit(self, X, y):
        # 1. Dopasuj CSP
        self.csp.fit(X[y == 0], X[y == 1])

        # 2. Transformuj dane CSP
        X_csp = self.csp.transform(X)
        if self.csp_components is not None:
            X_csp = X_csp[:, self.csp_components, :]

        # 3. Oblicz spektrogram
        Sxx = self.spectrogram.transform(X_csp, f_range=self.f_range, t_range=self.t_range)

        # 4. Spłaszcz dane
        X_flat = Sxx.reshape(Sxx.shape[0], -1)

        # 5. Dopasuj model klasyfikujący
        self.classifier.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        # 1. Zastosuj wytrenowany CSP
        X_csp = self.csp.transform(X)
        if self.csp_components is not None:
            X_csp = X_csp[:, self.csp_components, :]

        # 2. Oblicz spektrogram
        Sxx = self.spectrogram.transform(X_csp, f_range=self.f_range, t_range=self.t_range)

        # 3. Spłaszcz dane
        X_flat = Sxx.reshape(Sxx.shape[0], -1)

        # 4. Predykcja
        return self.classifier.predict(X_flat)

    def predict_proba(self, X):
        if hasattr(self.classifier, 'predict_proba'):
            X_csp = self.csp.transform(X)
            Sxx = self.spectrogram.transform(X_csp)
            X_flat = Sxx.reshape(Sxx.shape[0], -1)
            return self.classifier.predict_proba(X_flat)
        else:
            raise NotImplementedError("The classifier does not support predict_proba.")

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
