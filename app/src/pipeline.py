from helpers.model import cnn_model
from constants import (
    SEED_IMAGES,
    SEED_LABEL
)

initial_model = cnn_model(
    seed_images=SEED_IMAGES,
    seed_labels=SEED_LABEL,
    visualization=True
)
