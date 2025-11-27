import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# ---------- center input dataset ----------
def center(X, P):
    """
    X: shape (P, N)  where P = #points, N = #dimensions
    return: centered X, shape same as X
    """
    X_means = np.mean(X, axis=0)                     # (N,)
    X_centered = X - np.repeat(X_means.reshape(1,-1), P, axis=0) # np.repeat
    return X_centered


# ---------- compute Principal Components ----------
def compute_pcs(X, lam, P):
    """
    X: centered X
    lam: small constant to stabilize covariance
    P: #points
    return: eigenvalues D, eigenvectors V (columns)
    """
    Cov = 1/(P-1) * np.dot(X.T, X) + lam * np.eye(X.shape[1]) # \frac{1}{P}XX^T
    D, V = np.linalg.eigh(Cov)         # eigh for symmetric matrix
    # reverse order (largest -> smallest)
    idx = np.argsort(D)[::-1]
    D = D[idx]
    V = V[:, idx]
    return D, V

def finddim(cum_var, threshold):
    """
    choose PCA k dim based on threshold:
        e.g. threshold = 0.8/0.9/0.95/0.99
    """
    return np.argmax(cum_var >= threshold) + 1

def evaluate_kmeans(X, y_true, k=10, seed=0):
    """
    X      : data (P, D)
    y_true : true labels, shape (P,)
    return : acc, nmi, ari
    """
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    y_pred = kmeans.fit_predict(X)

    # NMI & ARI
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    # Accuracy 
    C = contingency_matrix(y_true, y_pred)  # shape: (n_classes, k)
    row_ind, col_ind = linear_sum_assignment(-C)  
    acc = C[row_ind, col_ind].sum() / C.sum()

    return acc, nmi, ari