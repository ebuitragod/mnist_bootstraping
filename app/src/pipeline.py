from helpers.model import cnn_model
from helpers.embeddings import generation, projection_2d
from constants import (
    SEED_IMAGES,
    SEED_LABEL,
    TRAIN_IMAGES,
)


history_model, initial_model = cnn_model(
    seed_images=SEED_IMAGES,
    seed_labels=SEED_LABEL,
    visualization=True
)

train_embeddings, seed_embeddings = generation(
    initial_model,
    TRAIN_IMAGES,
    SEED_IMAGES   
)

embeddings_2d = projection_2d(
    train_embeddings
)
