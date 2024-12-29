import numpy as np

# 等频离散化
def equal_frequency_binning(X, n_bins=10):
    X_binned = np.copy(X)
    for i in range(X.shape[1]):
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(X[:, i], quantiles)
        bins = np.unique(bins)
        if len(bins) - 1 < n_bins:
            n_actual_bins = len(bins) - 1
            # print(f"Feature {i}: Reduced number of bins to {n_actual_bins} due to duplicate edges.")
        X_binned[:, i] = np.digitize(X[:, i], bins) - 1
    return X_binned