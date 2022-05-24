import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def euclidean_distance(centroid, row, column_count):
    distance = 0.0
    
    for i in range(column_count): distance += (centroid[i] - row[i])**2
    
    return np.sqrt(distance)


def kmeans_firefly(X, k): 
  cluster = np.zeros(X.shape[0])
  
  random_indices = np.random.choice(len(X), size=k, replace=False)
  centroids = X[random_indices, :]
  
  column_count = len(X[0])
  
  c_rate = 1
  
  diff = 1
  while diff:

    # for each observation
    for i, row in enumerate(X):

      mn_dist = float('inf')
      # dist of the point from all centroids
      for idx, centroid in enumerate(centroids):
        d = euclidean_distance(centroid, row, column_count)

        # store closest centroid 
        if mn_dist > d:
          mn_dist = d
          cluster[i] = idx

    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    
    for i, row in enumerate(X):

      mn_dist = float('inf')
      # dist of the point from all centroids
      for idx, centroid in enumerate(centroids):
        if(not np.array_equal(centroid, row)): 
            d = euclidean_distance(centroid, row, column_count)
        size = pd.DataFrame(X).groupby(by=cluster).size()[idx]
     
        
        gl = size / d

        # store closest centroid 
        if mn_dist < gl:
          mn_dist = d
          cluster[i] = idx

    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    

    # if centroids are same then leave
    c_rate += 1
    if np.count_nonzero(centroids-new_centroids) == 0:
      diff = 0
    else:
      centroids = new_centroids
  return centroids, cluster


def print_metrics(X, cluster, ret=False):

    if ret:
      return {
        "silhouette_score": silhouette_score(X, cluster),
        "davies_bouldin_score": davies_bouldin_score(X, cluster),
        "calinski_harabasz_score": calinski_harabasz_score(X, cluster)
      }

    print(silhouette_score(X, cluster))
    print(davies_bouldin_score(X, cluster))
    print(calinski_harabasz_score(X, cluster))


def plus_plus(ds, k, random_state=42):
    np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)


def enhanced_kmeans_firefly(X, k): 
  cluster = np.zeros(X.shape[0])
  
  centroids = plus_plus(X, k)
  column_count = len(X[0])
  c_rate = 1

  diff = 1
  while diff:

    # for each observation
    for i, row in enumerate(X):

      mn_dist = float('inf')
      # dist of the point from all centroids
      for idx, centroid in enumerate(centroids):
        d = euclidean_distance(centroid, row, column_count)

        # store closest centroid 
        if mn_dist > d:
          mn_dist = d
          cluster[i] = idx

    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    
    for i, row in enumerate(X):

      mn_dist = float('inf')
      # dist of the point from all centroids
      for idx, centroid in enumerate(centroids):
          
        if(not np.array_equal(centroid, row)): 
            d = euclidean_distance(centroid, row, column_count)
        
        size = pd.DataFrame(X).groupby(by=cluster).size()[idx]
     
        
        gl = size / d

        # store closest centroid 
        if mn_dist < gl:
          mn_dist = d
          cluster[i] = idx

    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    
    c_rate += 1
    # if centroids are same then leave
    if np.count_nonzero(centroids-new_centroids) == 0:
      diff = 0
    else:
      centroids = new_centroids
  return centroids, cluster


def enhanced_kmeans_firefly_algo(X_PCA, X_train):
    n = 5
    
    sil_score = float('inf')
    
    centroids_out = []
    cluster_out = []
    
    for i in range(2,10):
        centroids, cluster = enhanced_kmeans_firefly(X_PCA, i)
        temp_sil_score_k = calinski_harabasz_score(X_train, cluster)
        
        # print("k=",i,"| CHI = ",temp_sil_score_k)
        
        if(temp_sil_score_k > sil_score or sil_score == float('inf')):
            k = i
            sil_score = temp_sil_score_k
            centroids_out = centroids
            cluster_out = cluster

    return centroids_out, cluster_out, k, X_train


def run(data, orig, n=0.99, k=6):

  X_train = data.values

  X_std = StandardScaler().fit_transform(X_train)

  pca = PCA(n_components=n)
  pca.fit(X_std)
  X_PCA = pca.transform(X_std)

  # X_PCA = X
  _d = {
    "enhanced": None,
    "normal": None,
  }

  centroids_ekfa, cluster_ekfa, k, X_train = enhanced_kmeans_firefly_algo(X_PCA, X_train)

  orig["CLUSTER"] = cluster_ekfa

  _e = {}
  _e["cluster_data"] = json.loads(orig.to_json(orient="records"))
  _e["metrics"] = print_metrics(X_train, cluster_ekfa, ret=True)
  _e["summary"] = json.loads(
      orig[["REGION", "CITY", "CLUSTER"]].to_json(orient="records"))
  _e["K"] = k
  _d["enhanced"] = _e

  orig.to_csv("enchanced.csv", index=False)

  centroids, cluster = kmeans_firefly(X_PCA, k)

  orig["CLUSTER"] = cluster

  _n = {}
  _n["cluster_data"] = json.loads(orig.to_json(orient="records"))
  _n["metrics"] = print_metrics(X_train, cluster, ret=True)
  _e["summary"] = json.loads(
      orig[["REGION", "CITY", "CLUSTER"]].to_json(orient="records"))
  _n["K"] = k
  _d["normal"] = _n

  orig.to_csv("normal.csv", index=False)

  return _d
