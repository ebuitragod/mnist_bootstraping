import numpy as np

from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from typing import Optional

def correct_by_neighbors(
    embeddings:np.ndarray,
    labels:np.ndarray,
    k:int = 15,
    min_agreement:float = 0.6
    ) -> np.ndarray:
    """
    It corrects the labels based on the labels with the minimum agreement and the k-nearest (euclideanly) neiboors.
            Args:
                embeddings:
                    Feature matrix
                labels: 
                    Current labels that will be transformed.
                k: 
                    Number of neighboors to be considered excluding itself.
                min_agreement: 
                    Minimum fraction in which the neighboors shall agree.
            Returns:
                pseudo_labels_corrected:
                    New array of corrected labels
    """
    nbrs = NearestNeighbors(
        n_neighbors=k+1, 
        metric='euclidean'
        ).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings) 

    new_labels = labels.copy()
    changes = 0

    for i in range(len(labels)):
        neighbor_indices = indices[i][1:]
        neighbor_labels = labels[neighbor_indices]
        counter = Counter(neighbor_labels)
        most_common_label, count = counter.most_common(1)[0]
        agreement = count / k

        if agreement >= min_agreement and most_common_label != labels[i]:
            new_labels[i] = most_common_label
            changes += 1

    print(f"Neighboors: {changes} changed labels (from {len(labels)})")
    return new_labels


def self_training(
    embeddings:np.ndarray,
    labels:np.ndarray,
    clf: Optional[LogisticRegression]=None,
    max_iter:int = 3,
    threshold:float = 0.9,
    )-> np.ndarray:
    """
    It is the process of iterative self_training to refine the labels using a classifier.
            Args:
                embeddings:
                    Features matrix
                labels:
                    it is intended to be pseudo_labels_corrected to be modified.
                clf:
                    Classifier that is intended to be used, LogisticRegression is optional.
                max_iter:
                    Maximum number of iteration for self-training.
                threshold:
                    Trust threshold to accept a label. It must be a number in (0,1)
            Returns: 
                pseudo_labels_final:
                    It returns a final version of pseudo-labels.
    """
    if clf is None:
        clf = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
            )

    current_labels = labels.copy()
    total_changes = 0

    for iteration in range(max_iter):
        clf.fit(
            embeddings,
            current_labels
            )
        probs = clf.predict_proba(
            embeddings
            )
        preds = clf.predict(
            embeddings
            )
        max_probs = np.max(
            probs,
            axis=1
            )
        high_conf = max_probs > threshold
        changes = (
            preds != current_labels
            ) & high_conf
        current_labels[changes] = preds[changes]
        n_changes = np.sum(changes)
        total_changes += n_changes
        print(f"Iteration {iteration+1}: {n_changes} changes")

        if n_changes == 0:
            break

    print(f"Self‑training total: {total_changes} accumulated changes")
    return current_labels

