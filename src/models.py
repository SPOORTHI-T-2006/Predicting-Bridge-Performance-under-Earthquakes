"""Model zoo. 'Generative' models (they model p(x | class)) are the ones the paper proposed to try next."""
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Generative classifier: one Gaussian Mixture per class + class priors, Bayes rule for prediction.

    A class with few distinct samples automatically gets fewer mixture components
    (at least `min_per_comp` distinct samples per component), so tiny classes don't overfit.
    """

    def __init__(self, n_components=2, reg_covar=1e-2, min_per_comp=10, random_state=0):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.min_per_comp = min_per_comp
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mix_, log_prior = [], []
        for c in self.classes_:
            Xc = X[y == c]
            distinct = len(np.unique(Xc, axis=0))
            k = int(max(1, min(self.n_components, distinct // self.min_per_comp)))
            self.mix_.append(GaussianMixture(k, covariance_type="full", reg_covar=self.reg_covar,
                                             random_state=self.random_state).fit(Xc))
            log_prior.append(np.log(len(Xc) / len(X)))
        self.log_prior_ = np.array(log_prior)
        return self

    def predict_proba(self, X):
        ll = np.column_stack([m.score_samples(X) for m in self.mix_]) + self.log_prior_
        return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MagnitudeOnly(BaseEstimator, ClassifierMixin):
    """Sanity-check baseline: logistic regression on earthquake magnitude alone (column 1)."""

    def fit(self, X, y):
        self.lr_ = LogisticRegression(max_iter=2000).fit(X[:, [1]], y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(X[:, [1]])

    def predict(self, X):
        return self.lr_.predict(X[:, [1]])


def get_models():
    """name -> factory. First four rows of GENERATIVE are the 'other generative models' of the future work."""
    return {
        "MagnitudeOnly": lambda: MagnitudeOnly(),  # baseline, not a real contender
        "LogReg": lambda: LogisticRegression(max_iter=2000),
        "KNN(k=2)": lambda: KNeighborsClassifier(n_neighbors=2),
        "KNN(k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
        "QDA": lambda: QuadraticDiscriminantAnalysis(),
        "QDA-reg": lambda: QuadraticDiscriminantAnalysis(reg_param=0.2),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "GaussianNB": lambda: GaussianNB(),
        "GMM": lambda: GMMClassifier(),
    }


GENERATIVE = {"QDA", "QDA-reg", "LDA", "GaussianNB", "GMM"}
