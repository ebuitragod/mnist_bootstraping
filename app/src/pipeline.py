from helpers.model import cnn_model
from helpers.embeddings import embedding_generation
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

train_embeddings, seed_embeddings = embedding_generation(
    initial_model,
    TRAIN_IMAGES,
    SEED_IMAGES   
)
