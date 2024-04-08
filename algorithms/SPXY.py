import numpy as np
from sklearn import metrics

"""SPXY concerns the independent and dependent variables in the distance
calculation. 
"""

def split(X, X_values, y, train_size):
    n = len(X)
    k = n*train_size
    
    x_dist = metrics.pairwise_distances(X_values, metric='euclidean', n_jobs=-1)
    x_dist = x_dist/np.max(x_dist)
    y_dist = metrics.pairwise_distances(y, metric='euclidean', n_jobs=-1)
    y_dist = y_dist/np.max(y_dist)

    dist = x_dist + y_dist

    full_set = set(list(range(n)))
    # Select two samples with largest distance
    idx_0, idx_1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    selected = set([idx_0, idx_1])
    k -= 2
    m_j = idx_1

    # Iteratively select samples with the largest minimum distance
    while k > 0 and len(selected) < n:
        minimum_dist = 0
        for j in range(n):
            if j not in selected:
                j_minimum = min([dist[j][i] for i in selected])
                if j_minimum > minimum_dist:
                    m_j = j
                    minimum_dist = j_minimum
        selected.add(m_j)
        k -= 1

    # Return training and testing set
    x_train = [X[s] for s in selected]
    x_test = [X[s] for s in full_set-selected]
    y_train = [y[s] for s in selected]
    y_test = [y[s] for s in full_set-selected]

    # Return training and testing set
    return x_train, x_test, y_train, y_test