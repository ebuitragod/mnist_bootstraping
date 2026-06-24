from helpers import load_data as ld

# datasets
TRAINSET = ld.load_mnist_normalized(train=True)
TESTSET = ld.load_mnist_normalized(train=False)
SEEDSET = TESTSET

# images
TRAIN_IMAGES = ld.images_extractor(TRAINSET)
TEST_IMAGES = ld.images_extractor(TESTSET)
SEED_IMAGES = TEST_IMAGES.reshape((-1, 28, 28, 1))

# labels
TRAIN_LABELS = ld.labels_extractor(TRAINSET) # groundtruth
TEST_LABELS = ld.labels_extractor(TESTSET)
SEED_LABEL = TEST_LABELS.copy()

# parameters to be used in the pipeline
CONFIG = {
    'cnn': {
        'epochs': 20,
        'batch_size': 64,
        'drop_prob': 0.5,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'visualization': True,
    },
    'clustering': {
        'n_clusters': 10,
        'sample_size': 3000,
        'visualization': True,
    },
    'label_correction': {
        'k': 15,
        'min_agreement': 0.6,
    },
    'self_training': {
        'max_iter': 3,
        'threshold': 0.9,
    },
}
