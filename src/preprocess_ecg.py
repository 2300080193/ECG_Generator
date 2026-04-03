import numpy as np


def normalize(X):
    # Scale to 0–1
    X = (X - X.min()) / (X.max() - X.min())

    # Scale to -1 to 1 (important for Tanh)
    return X * 2 - 1