import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self._metric = metric        

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        s = np.zeros(X.shape[0])
        distances = cdist(X, X, self._metric)
        for i in range(X.shape[0]):
            a = self._calculate_a(distances, y, i)
            b = self._calculate_b(distances, y, i)
            s[i] = (b - a)/np.max([a, b])
        return s

    def _calculate_a(self, distances, y, i):
        """
        Calculate the intra cluster distance for a data point
        """
        distances = distances[i,y == y[i]]
        return np.sum(distances)/(np.sum(y == y[i]) - 1)

    def _calculate_b(self, distances, y, i):
        """
        Calculate the inter cluster distance for a data point
        """
        inter_distances = np.ones(np.max(y)) * np.inf
        for j in range(np.max(y)):
            if j != y[i]:
                inter_distances[j] = np.sum(distances[i,y == j])/np.sum(y == j)
        return np.min(inter_distances)

