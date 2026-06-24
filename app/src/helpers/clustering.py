import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from typing import Tuple, Dict

def get_kmeans_pseudo_labels(
    embeddings: np.ndarray,
    n_clusters:int = 10,
) -> Tuple[np.ndarray, KMeans]:
    """
    Applies KMeans algorith to the embeddings to get the cluster-labels.
            Args:
                embeddings:
                    Feature matrix
                n_clusters:
                    Number of clusters
            Returns:
                pseudo-labels:
                    Cluster labels array.
                kmeans:
                    KMEAN adjusted model with cluster_centers attributes
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
        )
    pseudo_labels = kmeans.fit_predict(embeddings)
    return pseudo_labels, kmeans


def get_cluster_class_from_logistic_regression(
    seed_embeddings:np.ndarray,
    seed_labels:np.ndarray,
    cluster_centers:np.ndarray,
) -> np.ndarray:
    """
    Assigning to each center cluster to a class using the LogisticRegresion trained over a seedset.
        1. Train embedding classifier on seed set
        2. Predict class for each cluster center
        
        Args:
            seed_embeddings:
                Embeddings matrix of seedset
            seed_labels:
                seed true labels from seedset
            cluster_centers:
                Matrix of cluster centers
        Returns:
            cluster_classes:
                array with predicted class for cluster_center
    """
    clf = LogisticRegression(
        max_iter=1000, 
        multi_class='multinomial',
        solver='lbfgs'
        )
    clf.fit(seed_embeddings, seed_labels)

    return clf.predict(cluster_centers)

    
def get_sample_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 10,
    sample_size: int = 3000
):
    """
    Visualization of a 2d-embeddings with TSNE, coloured by label. 
            Args:
                embeddings:
                    Feature matrix
                labels:
                    label array
                n_clusters:
                    number of clusters
                sample_size:
                    sample size
            Returns:
                a saved figure in `app/output/kmeans_labels_visualize.png`
    """
    
    if embeddings.shape[0] > sample_size:
        indices = np.random.choice(
            embeddings.shape[0], 
            sample_size,
            replace=False
            )
        X_sample = embeddings[indices]
        sample_labels = labels[indices]
    else:
        X_sample = embeddings
        sample_labels = labels

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
        )
    X_vis = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X_vis[:,0], 
        X_vis[:,1], 
        c = sample_labels, 
        cmap = 'tab10',
        s = 5,
        alpha = 0.6
        )
    plt.colorbar(scatter)
    plt.title(f"KMeans Clustering (k={n_clusters})")
    plt.savefig('../output/kmeans_labels_visualize.png', dpi=150)
    plt.show()
    
    
def get_kmeans_labels_and_visualize(
    embeddings, 
    n_clusters=10, 
    sample_size=3000,
    visualization: bool = True,
    ):
    """
    Perform KMeans clustering on full embeddings and visualize a sample.
    Returns cluster labels for the full dataset.
            Args:
                embeddings:
                    Feature matrix
                n_clusters:
                    Number of clusters
                visualization:
                    If true, saved a figure in `app/output/kmeans_labels_visualize.png`
            Returns:
                pseudo-labels:
                    Cluster labels array.
    """
    pseudo_labels, _ = get_kmeans_pseudo_labels(
        embeddings,
        n_clusters,
    )
    if visualization:
        get_sample_visualization(
            embeddings,
            pseudo_labels,
            n_clusters = n_clusters,
            sample_size=sample_size
        )
    return pseudo_labels


def assign_initial_labels_from_logistic_regression(
    train_embeddings:np.ndarray,
    seed_embeddings:np.ndarray,
    seed_labels:int,
    n_clusters:int = 10
    ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Cluster training embeddings and assign each cluster to a digit class
    using a classifier trained on seed embeddings.
            Args:
                train_embeddings:
                    Embeddings matrix over trainset
                seed_embeddings:
                    Embeddings matrix over seedset
                seed_labels:
                    True labels over seedset
                n_clusters:
                    Number or clusters
            Returns:
                pseudo_labels:
                    Arry with pseudo-labels
                cluster_to_class:
                    Dictionary that maps index-cluster over predicted class from cls
    """
    cluster_labels, kmeans = get_kmeans_pseudo_labels(
            train_embeddings,
            n_clusters,
        )
    cluster_centers = kmeans.cluster_centers_

    cluster_classes = get_cluster_class_from_logistic_regression(
        seed_embeddings,
        seed_labels, # pyright: ignore[reportArgumentType]
        cluster_centers,
    )
    cluster_to_class = { i: 
        int(cluster_classes[i]) 
        for i in range(n_clusters)
    }
    
    pseudo_labels = np.array(
        [
            cluster_to_class[cl]
            for cl in cluster_labels
        ]
    )
    print("Train embeddings shape:", train_embeddings.shape) 
    print("Seed embeddings shape:", seed_embeddings.shape)  
    print("Pseudo labels shape:", pseudo_labels.shape)   

    return pseudo_labels, cluster_to_class
