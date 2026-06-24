
from constants import (
    SEED_IMAGES,
    SEED_LABEL,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)
from helpers.clustering import get_kmeans_labels_and_visualize, assign_initial_labels_from_logistic_regression
from helpers.embeddings import generation, projection_2d
from helpers.label_correction import correct_by_neighbors, self_training
from helpers.model import cnn_model
from sklearn.metrics import accuracy_score


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

pseudo_labels = get_kmeans_labels_and_visualize(
    embeddings_2d, 
    n_clusters=10,
    )


pseudo_labels, cluster_to_class = assign_initial_labels_from_logistic_regression(
    train_embeddings, 
    seed_embeddings, 
    seed_labels = SEED_LABEL, # pyright: ignore[reportArgumentType]
    n_clusters=10
    )

pseudo_labels_corrected = correct_by_neighbors(
    train_embeddings,
    pseudo_labels,
    k=15, 
    min_agreement=0.6
    )

# only informative
acc_after_neighbors = accuracy_score(
    TRAIN_LABELS,
    pseudo_labels_corrected
    )
print(f"Accuracy after correcting by neighboors: {acc_after_neighbors:.4f}")

pseudo_labels_final = self_training(
    train_embeddings,
    pseudo_labels_corrected,
    max_iter=3,
    threshold=0.9
)

acc_final = accuracy_score(
    TRAIN_LABELS,
    pseudo_labels_final
    )
print(f"\nCombined (final) accuracy: {acc_final:.4f}")

print("\nSummary of corrections:")
print(f"  - Initial labels:                        {accuracy_score(TRAIN_LABELS, pseudo_labels):.4f}")
print(f"  - After neighboors:                      {acc_after_neighbors:.4f}")
print(f"  - After self‑training (final accuracy):  {acc_final:.4f}")
