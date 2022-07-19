import numpy as np
from sklearn.neighbors import KDTree


def relief_f(X, y, n_neighbors=100):
    feature_scores = np.zeros(X.shape[1])
    tree = KDTree(X)

    for source_index in range(X.shape[0]):
        distances, indices = tree.query(X[source_index].reshape(1, -1), k=n_neighbors + 1)

        # Nearest neighbor is self, so ignore first match
        indices = indices[0][1:]

        # Create a binary array that is 1 when the source and neighbor
        #  match and -1 everywhere else, for labels and features..
        labels_match = np.equal(y[source_index], y[indices]) * 2. - 1.
        features_match = np.equal(X[source_index], X[indices]) * 2. - 1.

        # The change in feature_scores is the dot product of these  arrays
        feature_scores += np.dot(features_match.T, labels_match)

    return feature_scores
