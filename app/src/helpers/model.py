from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


learn_rate:float = 0.001

def visualization_training_validation_accuracy(
    history_model: tf.keras.callbacks.History
):
    """
    It generates a plot in `app/output/accuracy_plot.png`
            Args:
                history_model:
                    Keras model with history to be visualized.
    """
    
    plt.plot(
        history_model.history['accuracy'], 
        c = "b", 
        label = 'Train Accuracy'
        )
    plt.plot(
        history_model.history['val_accuracy'],
        c = "g",
        label = 'Validation Accuracy'
        )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.grid()
    plt.savefig('../output/accuracy_plot.png', dpi=150)
    plt.show()



def cnn_model(
    seed_images:np.ndarray,
    seed_labels:np.ndarray,
    drop_prob: float = 0.5,
    epochs:int = 20,  
    batch_size:int = 64,
    validation_split:float = 0.2,
    learning_rate:float = 0.001,
    visualization:bool = True
) -> tf.keras.callbacks.History:
    """
    It builds and trains a CNN for image classification.
            Args:
                seed_images:
                    Training seed images (from testset)
                seed_labels:
                    Training seed labels (from testset)
                drop_prob:
                    Dropout probability in dropouts layers.
                epochs:
                    Number of training epochs.
                batch_size:
                    Size of the batch.
                validation_split:
                    Fraction of training data (testset) to be used as validation. It must be a number in (0,1)
                learning_rate:
                    Learning rate to for Adam optimizer
                visualization:
                    If true, generates a plot in `app/output/accuracy_plot.png`
            Returns:
                history model with epoch training
    """
    
    model = models.Sequential([
        layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3),
            activation='relu', 
            ),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drop_prob),
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3), 
            activation='relu'
            ),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drop_prob),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_prob),
        layers.Dense(10, activation='softmax')
    ])  

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
        )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
        )
    
    history = model.fit(
        seed_images, 
        seed_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
        )

    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    if visualization:
        visualization_training_validation_accuracy(
            history
        )
    return history
