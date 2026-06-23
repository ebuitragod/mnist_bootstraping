import numpy as np
import torch

from torchvision import datasets, transforms

DATA_DIR:str = "../output/data"


def images_extractor(
    dataset: datasets.MNIST
    ) -> np.ndarray:
    """
    Extracts images from a pytorch dataset. 
        Args:
            dataset: 
                mean to be either testset or trainsset from MNIST
        Returns:
            Numpy array with the image information
    """
    return torch.stack(
        [
            dataset[i][0]
            for i in range(len(dataset))
        ]
    ).numpy()


def labels_extractor(
    dataset: datasets.MNIST
    )->np.ndarray:
    """
    Extracts labels from a pytorch dataset MNIST. 
        Args:
            dataset: 
                mean to be either testset or trainsset from MNIST
        Returns:
            Numpy array with the label information
    """
    return torch.tensor(
        [
            dataset[i][1] 
            for i in range(len(dataset))
        ]
    ).numpy()


def load_mnist_normalized(
    data_dir:str = DATA_DIR,
    train: bool = False
) -> datasets.MNIST:
    """
    Downloading MNist data set
            Args:
                data_dir: 
                    where the data will be store locally
                train: 
                    boolean if True downloads trainset, otherwise testset. 
            Returns:
                MNIST dataset. If train = false, then it returns `testset`, otherwise `traintest`.
    """
    transform = transforms.Compose([
            transforms.ToTensor()
        ])    
    dataset = datasets.MNIST(
        root = data_dir,
        train = False,
        download = True,
        transform = transform
    )
    dataset_images = images_extractor(dataset)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (dataset_images.mean().item(),),
            (dataset_images.std().item(),)
            )
    ])
    
    return datasets.MNIST(
        root = data_dir,
        download = True,
        train = train,
        transform = transform
    )
