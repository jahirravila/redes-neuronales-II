"""Implementaciones NumPy reutilizables (Notebook 01): activaciones, Perceptrón,
red de una capa (SingleLayerNet) y MLP con backpropagation."""
import numpy as np

SEED = 42


def step(z):        return (z >= 0).astype(float)
def sigmoid(z):     return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def d_sigmoid(z):   s = sigmoid(z); return s * (1 - s)
def tanh(z):        return np.tanh(z)
def d_tanh(z):      return 1 - np.tanh(z) ** 2
def relu(z):        return np.maximum(0, z)
def d_relu(z):      return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))   # estabilidad numérica
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {"sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh), "relu": (relu, d_relu)}


class Perceptron:
    """Perceptrón de Rosenblatt con activación escalón."""
    def __init__(self, lr=0.1, epochs=50, seed=SEED):
        self.lr, self.epochs, self.rng = lr, epochs, np.random.default_rng(seed)

    def fit(self, X, y):
        self.w = self.rng.normal(0, 0.01, X.shape[1]); self.b = 0.0
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for i in self.rng.permutation(len(X)):
                update = self.lr * (y[i] - self.predict(X[i:i+1])[0])
                self.w += update * X[i]; self.b += update
                errors += int(update != 0)
            self.errors_.append(errors)
            if errors == 0:          # convergió
                break
        return self

    def decision_function(self, X): return X @ self.w + self.b
    def predict(self, X):           return step(self.decision_function(X))
    def score(self, X, y):          return (self.predict(X) == y).mean()

class SingleLayerNet:
    """Red de una capa: una neurona sigmoide + entropía cruzada + descenso de gradiente (batch)."""
    def __init__(self, lr=0.5, epochs=2000, l2=0.0, seed=SEED):
        self.lr, self.epochs, self.l2, self.rng = lr, epochs, l2, np.random.default_rng(seed)

    def fit(self, X, y):
        m = len(X)
        self.w = self.rng.normal(0, 0.01, X.shape[1]); self.b = 0.0
        self.loss_ = []
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)                     # forward
            self.loss_.append(bce(y, p) + 0.5 * self.l2 * np.sum(self.w**2) / m)
            dz = (p - y) / m                                      # backward
            self.w -= self.lr * (X.T @ dz + self.l2 * self.w / m)
            self.b -= self.lr * dz.sum()
        return self

    def predict_proba(self, X): return sigmoid(X @ self.w + self.b)
    def predict(self, X):       return (self.predict_proba(X) >= 0.5).astype(float)
    def score(self, X, y):      return (self.predict(X) == y).mean()

def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


class MLP:
    """Perceptrón multicapa en NumPy con backpropagation.

    layers: p. ej. [n_features, 16, 8, 1]. Salida sigmoide si la última capa tiene 1 neurona
    (clasificación binaria), softmax si tiene >1 (multiclase).
    """
    def __init__(self, layers, hidden="relu", lr=0.05, epochs=500, batch_size=32,
                 momentum=0.9, l2=0.0, seed=SEED, verbose=0):
        self.layers, self.hidden = layers, hidden
        self.g, self.dg = ACTIVATIONS[hidden]
        self.lr, self.epochs, self.bs = lr, epochs, batch_size
        self.beta, self.l2, self.verbose = momentum, l2, verbose
        self.rng = np.random.default_rng(seed)
        self.binary = layers[-1] == 1
        self._init_params()

    def _init_params(self):
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            scale = np.sqrt(2.0 / n_in) if self.hidden == "relu" else np.sqrt(1.0 / n_in)  # He / Xavier
            self.W.append(self.rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ---------- forward ----------
    def forward(self, X):
        A, cache = X, [(None, X)]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            last = l == len(self.W) - 1
            A = (sigmoid(Z) if self.binary else softmax(Z)) if last else self.g(Z)
            cache.append((Z, A))
        return A, cache

    def loss(self, Y, A):
        data = bce(Y, A) if self.binary else -np.mean(np.sum(Y * np.log(np.clip(A, 1e-12, 1)), axis=1))
        reg = 0.5 * self.l2 * sum(np.sum(W**2) for W in self.W) / len(Y)
        return data + reg

    # ---------- backward ----------
    def backward(self, Y, cache):
        m = len(Y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        delta = cache[-1][1] - Y                                   # δ^[L]
        for l in reversed(range(len(self.W))):
            A_prev = cache[l][1]
            gW[l] = A_prev.T @ delta / m + self.l2 * self.W[l] / m
            gb[l] = delta.sum(axis=0, keepdims=True) / m
            if l > 0:
                delta = (delta @ self.W[l].T) * self.dg(cache[l][0])  # δ^[l-1]
        return gW, gb

    def _step(self, gW, gb):
        for l in range(len(self.W)):
            self.vW[l] = self.beta * self.vW[l] - self.lr * gW[l]; self.W[l] += self.vW[l]
            self.vb[l] = self.beta * self.vb[l] - self.lr * gb[l]; self.b[l] += self.vb[l]

    def _prep_y(self, y):
        y = np.asarray(y)
        if self.binary: return y.reshape(-1, 1).astype(float)
        return np.eye(self.layers[-1])[y.astype(int)]

    def fit(self, X, y, X_val=None, y_val=None):
        Y = self._prep_y(y)
        self.history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        for ep in range(self.epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), self.bs):
                bi = idx[s:s + self.bs]
                A, cache = self.forward(X[bi])
                self._step(*self.backward(Y[bi], cache))
            A, _ = self.forward(X)
            self.history["loss"].append(self.loss(Y, A)); self.history["acc"].append(self.score(X, y))
            if X_val is not None:
                Av, _ = self.forward(X_val)
                self.history["val_loss"].append(self.loss(self._prep_y(y_val), Av))
                self.history["val_acc"].append(self.score(X_val, y_val))
            if self.verbose and (ep + 1) % self.verbose == 0:
                msg = f"época {ep+1:4d} | pérdida {self.history['loss'][-1]:.4f} | acc {self.history['acc'][-1]:.3f}"
                if X_val is not None: msg += f" | val_pérdida {self.history['val_loss'][-1]:.4f} | val_acc {self.history['val_acc'][-1]:.3f}"
                print(msg)
        return self

    def predict_proba(self, X): return self.forward(X)[0]
    def predict(self, X):
        A = self.predict_proba(X)
        return (A[:, 0] >= 0.5).astype(float) if self.binary else A.argmax(axis=1)
    def score(self, X, y): return (self.predict(X) == np.asarray(y)).mean()
