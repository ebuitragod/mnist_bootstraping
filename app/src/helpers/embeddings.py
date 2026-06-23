

import tensorflow as tf


def embedding_generation(
    initial_model,
    train_images,
    seed_images,    
):
    """
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
