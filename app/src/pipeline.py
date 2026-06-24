from constants import (
    SEED_IMAGES,
    SEED_LABEL,
    TRAIN_IMAGES,
    TRAIN_LABELS,
    CONFIG,
)
from helpers.clustering import get_kmeans_labels_and_visualize, assign_initial_labels_from_logistic_regression
from helpers.embeddings import generation, projection_2d
from helpers.label_correction import correct_by_neighbors, self_training
from helpers.model import cnn_model

import logging
import sys

from sklearn.metrics import accuracy_score


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_pipeline(config: dict): #-> Tuple[np.ndarray, Any]:
    """
    Pipeline of generation of pseudo-labels and training.
        1. Training a CNN models with seedset
        2. Extract embeddings
        3. 2D projection 
        4. Generation of initial-pseudo-lables
        5. Assigning lables with LogisticRegresion over clusters
        6. Neighboor correcting
        7. Selftraining
        8. Final evaluation
        9. Summary
        
        to run the pipeline you shall:
            `python pipeline.py --n_clusters 10 --max_iter 5 --threshold 0.90`
    """
    logger.info("="*30)
    logger.info("... Runing pipeline ...")
    logger.info("="*30)
    
    _, initial_model = cnn_model(
        seed_images=SEED_IMAGES,
        seed_labels=SEED_LABEL,
        visualization=True
    )
    logger.info("CNN models trained successfully.")
    
    logger.info("Extracting embeddings...")
    train_embeddings, seed_embeddings = generation(
        initial_model,
        TRAIN_IMAGES,
        SEED_IMAGES   
    )
    logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"Seed embeddings shape: {seed_embeddings.shape}")
    
    logger.info("Embedding projection to 2D with t-SNE...")
    embeddings_2d = projection_2d(
        train_embeddings,
        sample_size=config['clustering']['sample_size'],
        visualization=config['clustering']['visualization']
    )
    
    logger.info("Generating initial-pseudo-labels...")
    pseudo_labels = get_kmeans_labels_and_visualize(
        embeddings=embeddings_2d,
        n_clusters=config['clustering']['n_clusters'],
        sample_size=config['clustering']['sample_size'],
        visualization=config['clustering']['visualization']
    )
    logger.info(f"Pseudo-etiquetas iniciales generadas: {pseudo_labels.shape}")
    
    logger.info("Assigning labels to clusters with LogisticRegression...")
    pseudo_labels, _ = assign_initial_labels_from_logistic_regression(
        train_embeddings=train_embeddings,
        seed_embeddings=seed_embeddings,
        seed_labels=SEED_LABEL, # pyright: ignore[reportArgumentType]
        n_clusters=config['clustering']['n_clusters']
    )
    logger.info(f"Pseudo-labels mapped: {pseudo_labels.shape}")
    
    logger.info("Correcting labels with near neighboors...")
    pseudo_labels_corrected = correct_by_neighbors(
        embeddings=train_embeddings,
        labels=pseudo_labels,
        k=config['label_correction']['k'],
        min_agreement=config['label_correction']['min_agreement']
    )

    # only informative
    acc_after_neighbors = accuracy_score(TRAIN_LABELS, pseudo_labels_corrected)
    logger.info(f"Accuracy after neighboors: {acc_after_neighbors:.4f}")
    
    logger.info("Selftraining for labeling refining...")
    pseudo_labels_final = self_training(
        embeddings=train_embeddings,
        labels=pseudo_labels_corrected,
        max_iter=config['self_training']['max_iter'],
        threshold=config['self_training']['threshold']
    )
    
    acc_final = accuracy_score(TRAIN_LABELS, pseudo_labels_final)
    logger.info(f"Final accuracy after selftraining: {acc_final:.4f}")
    
    # Accuracy
    initial_acc = accuracy_score(TRAIN_LABELS, pseudo_labels)
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 30)
    logger.info(f"  - Initial (clustering + Logistic Regresion): {initial_acc:.4f}")
    logger.info(f"  - Accuracy after neighbor correction:          {acc_after_neighbors:.4f}")
    logger.info(f"  - Final (after self-training):       {acc_final:.4f}")
    logger.info("=" * 60)

    return pseudo_labels, initial_model


if __name__=="__main__":
    try:
        pseudo_labels_final, model = run_pipeline(CONFIG)
        logger.info("Pipeline ran successfully.")
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        sys.exit(1)
