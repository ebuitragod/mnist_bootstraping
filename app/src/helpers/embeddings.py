

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE
from typing import Tuple, Any


def generation(
    initial_model: tf.keras.Model,
    train_images: np.ndarray,
    seed_images: np.ndarray,    
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracting embeddings from a pre-trained model to the training and seed images,
    by using the last dense layer
            Args:
                initial_model:
                    Pre-trained model to extract its representations.
                train_images:
                    Set of train images.
                seed_images:
                    set of seed images.
            Returns:
                train_embeddings: 
                    Embeddings from training set.
                seed_embeddings:
                    Embeddings from seed set.
    """
    layer_names = [
        layer.name 
        for layer in initial_model.layers
    ]
    print(layer_names)

    dense_layers = [
        layer.name 
        for layer in initial_model.layers 
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    print("Available Dense layzers:", dense_layers)
    selected_layer = dense_layers[-1]
    print(selected_layer)

    embedding_model = tf.keras.Model(
        inputs=initial_model.inputs, 
        outputs=initial_model.get_layer(selected_layer).output
        ).predict
    
    train_embeddings = embedding_model(
        train_images.reshape(-1,28,28,1)
        )

    seed_embeddings = embedding_model(
        seed_images.reshape(-1,28,28,1)
    )
    return train_embeddings, seed_embeddings


def projection_2d(
    train_embeddings: np.ndarray,
    sample_size: int = 3000,
    visualization: bool = True
) -> np.ndarray:
    """
    It prjects embedding of high dimension into a 2D space by using TSNE.
            Args:
                train_embeddings:
                    Embedding matrix from trainset.
                sample_size:
                    Sample size to use the projection.
                visualization:
                    If true, it will save a picture in `app/output/embeddings_2d.png`
            Returns:
                embeddings_2d with the 2-dimensional projection of the given embeddings. 
    """
    embeddings_2d =  TSNE(n_components=2).fit_transform(
        train_embeddings[:sample_size]
        )  

    if visualization:
        plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            cmap='tab10'
            )
        plt.colorbar()
        plt.savefig('../output/embeddings_2d.png', dpi=150)
        plt.show()
        
    return embeddings_2d


