import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.array([0, 1])
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.feature_neg_log_prob_ = {}

        for c in self.classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]

            if n_c == 0:
                self.class_log_prior_[c] = -np.inf
                theta = np.full(n_features, self.smoothing / (2 * self.smoothing))
            else:
                self.class_log_prior_[c] = np.log(n_c / n_samples)
                N_jc = np.sum(X_c, axis=0)
                theta = (N_jc + self.smoothing) / (n_c + 2 * self.smoothing)

            self.feature_log_prob_[c] = np.log(theta)
            self.feature_neg_log_prob_[c] = np.log(1 - theta)

    def predict(self, X):
        posteriors = np.zeros((X.shape[0], 2))
        for c in self.classes:
            if self.class_log_prior_[c] == -np.inf:
                posteriors[:, c] = -np.inf
            else:
                ll = X @ self.feature_log_prob_[c] + (1 - X) @ self.feature_neg_log_prob_[c]
                posteriors[:, c] = ll + self.class_log_prior_[c]
        return np.argmax(posteriors, axis=1)