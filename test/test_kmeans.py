from cluster import (
  KMeans,
  make_clusters)
import numpy as np

# Write your k-means unit tests here

def test_catch_zero():
    """ 
    Ensure KMeans does not initialize with k <= 0
    """
    try: 
        km = KMeans(k = 0)
        assert(False)
    except AssertionError:
        pass

def test_tightly_clustered():
    """
    Check that the clustering is successful on tightly-scaled data with 3 clusters and 600 data points
    """
    t_clusters, t_labels = make_clusters(scale=0.2, n = 600)
    km = KMeans(k = 3)
    km.fit(t_clusters)
    preds = km.predict(t_clusters)
    # Identify the sizes of true clusters
    n_zeros = np.sum(t_labels == 0)
    n_ones = np.sum(t_labels == 1)
    # Check that the true clusters each map to exactly one predicted cluster
    assert len(set(preds[0:n_zeros])) == 1
    assert len(set(preds[n_zeros:(n_zeros + n_ones)])) == 1
    assert len(set(preds[(n_zeros + n_ones):])) == 1
         
