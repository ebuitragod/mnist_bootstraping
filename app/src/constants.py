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
