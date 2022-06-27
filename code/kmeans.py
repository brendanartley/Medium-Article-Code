import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def euclidian_distance(p1, p2):
    return sum([(x-y)**2 for x,y in zip(p1, p2)])**(1/2)

def update_centroids(centroid_dict, cluster_dict, X):
    X_dim = len(X[0])

    for key in cluster_dict.keys():
        cluster_xs = [X[i] for i in cluster_dict[key]]

        if len(cluster_xs)!=0:
            new_cluster = np.mean(cluster_xs, axis=0)

        centroid_dict[key] = new_cluster
    return centroid_dict, cluster_dict

def update_clusters(centroid_dict, cluster_dict, X, initial=False):
    if initial == False:
        for key in cluster_dict.keys():
            cluster_dict[key] = []

    for i,x in enumerate(X):
        min_dist = np.inf
        min_cluster = None
        for c in cluster_dict.keys():
            current_dist = euclidian_distance(x, centroid_dict[c])
            if current_dist < min_dist:
                min_cluster = c
                min_dist = current_dist
        cluster_dict[min_cluster].append(i)
    
    return centroid_dict, cluster_dict

def check_dicts_identical(centroid_dict, last_centroid_dict):
        for x in centroid_dict.keys():
            if (centroid_dict[x] == last_centroid_dict[x]).all():
                print()
                continue
            else:
                return False
        return True

def train(centroid_dict, cluster_dict, X, epochs):
    for epoch in range(epochs):

        if epoch == 0:
            centroid_dict, cluster_dict = update_clusters(centroid_dict, cluster_dict, X, initial=True)
            continue

        last_centroid_dict = centroid_dict.copy()
        
        centroid_dict, cluster_dict = update_centroids(centroid_dict, cluster_dict, X)
        centroid_dict, cluster_dict = update_clusters(centroid_dict, cluster_dict, X)

        if check_dicts_identical(centroid_dict, last_centroid_dict):
            return centroid_dict, cluster_dict, epoch

        last_centroid_dict = centroid_dict.copy()
    return centroid_dict, cluster_dict, epoch

def plot_results(centroid_dict, cluster_dict, X, epoch):
    nrows=1
    ncols=1
    fig, ax = plt.subplots(figsize=(10, 6), nrows=nrows, ncols=ncols)
    
    ax1 = plt.subplot(nrows, ncols, 1)
    ax1.scatter(X[:, 0], X[:, 1], c='tab:blue')
    for k in centroid_dict.keys():
        ax1.scatter(centroid_dict[k][0], centroid_dict[k][1], c='red', s=100)
    ax1.set_title("K-Means Clustering - Converged on Epoch: {}".format(epoch))
    plt.show()

def numpy_implementation():

    n_clusters = 4
    n_samples = 300
    epochs = 100
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=0)
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.title("Data Points")
    plt.show()

    # --- Initializing Centroids as Random Points from Data --- 
    init_centroids = np.random.choice(range(n_samples), size=n_clusters, replace=False)

    centroid_dict = {}
    for i, c in zip(range(n_clusters), init_centroids):
        centroid_dict[i] = X[c]

    cluster_dict = {}
    for i in range(n_clusters):
        cluster_dict[i] = []

    # --- Training--- 
    centroid_dict, cluster_dict, epoch = train(centroid_dict, cluster_dict, X, epochs)

    # --- Plotting Results --- 
    plot_results(centroid_dict, cluster_dict, X, epoch)

def scikitlearn_implementation():
    n_clusters = 4
    n_samples = 300
    epochs = 100
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=0)

    # --- Training--- 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=epochs).fit(X)    
    print(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(X[:, 0], X[:, 1], c='tab:blue', s=25)
    for k in kmeans.cluster_centers_:
        ax1.scatter(k[0], k[1], c='red', s=50)
    ax1.set_title("Scikit-Learn K-means Results")
    plt.show()

if __name__ == '__main__':
    numpy_implementation()
    # scikitlearn_implementation()