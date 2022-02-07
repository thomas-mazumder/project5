import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100,
            n_restarts: int = 10):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
            n_restarts: int
                the number of random initializations the algorithm will conduct
        """
        assert(k > 0), "k must be greater than 0"
        self._k = k
        self._metric = metric
        self._tol = tol
        self._max_iter = max_iter
        self._centroids = None
        self._error = np.inf
        self._n_restarts = n_restarts
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # try n_restart initializations of the centroids
        for j in range(self._n_restarts):
            # initialize each centroid at a data point
            centroids = mat[np.random.choice(mat.shape[0], self._k, replace = False)]
            errors = np.zeros(self._max_iter)
            it = 0
            # run the EM algorithm for max_iter iterations
            for it in range(self._max_iter): 
                distances = cdist(mat, centroids, self._metric) # compute distances from each point to each centroid
                assignments = np.argmin(distances, axis = 1) # assign each point to closest centroid
                # update centroid positions
                for cluster in range(self._k):
                    centroids[cluster, :] = np.mean(mat[assignments == cluster,:], axis = 0)
                errors[it] = np.mean(np.linalg.norm(centroids[assignments,:] - mat, axis = 1))
                if it > 0:
                    if errors[it] - errors[it - 1] < self._tol: # optimization complete
                        break
            # if the lowest error was achieved, save the centroid positions and the error
            if errors[it] < self._error:
                self._centroids = centroids
                self._error = errors[it]


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        distances = cdist(mat, self._centroids, self._metric)
        return np.argmin(distances, axis = 1)

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids
