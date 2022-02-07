from cluster import (
  make_clusters,
  Silhouette)
import numpy as np

# write your silhouette score unit tests here

def test_good_labels():
    t_clusters, t_labels = make_clusters(scale=0.2, n = 600)
    sh = Silhouette()
    scores = sh.score(t_clusters, t_labels)
    assert np.all(scores >= -1)
    assert np.all(scores <= 1)
    assert np.mean(scores) > .9

def test_bad_labels():
    t_clusters, _ = make_clusters(scale=0.2, n = 600)
    bad_labels = np.array([0,1,2]*200)
    sh = Silhouette()
    scores = sh.score(t_clusters, bad_labels)
    assert np.all(scores >= -1)
    assert np.all(scores <= 1)
    assert np.mean(scores) < 0
