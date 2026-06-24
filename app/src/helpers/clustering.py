import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_kmeans_pseudo_labels(
    embeddings: np.ndarray,
    n_clusters:int = 10,
) -> np.ndarray:
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
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
        )
    return kmeans.fit_predict(embeddings)

    
def get_sample_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 10,
    sample_size: int = 3000
):
    """
    Visualization of a 2d-embeddings with TSNE, coloured by label. 
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
    pseudo_labels = get_kmeans_pseudo_labels(
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

