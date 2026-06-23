import tensorflow as tf
from torchvision import datasets, transforms
import numpy as np
import torch
from typing import Tuple



def load_mnist(data_dir:str ='./data')-> datasets.MNIST:
    """
    download MNIST if not already downloaded, and return the 
    training and test datasets as numpy arrays.
    
    Returns a tuple where:
    1 = Dataset MNIST
        Number of datapoints: 10000
        StandardTransform 
        
    But, we know that
    Media: 0.1325, Std: 0.3105
    """
    
    mean = 0.1325 
    std = 0.3105

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    testset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    return testset


def images_extractor(dataset):
    """
    Extrae las imágenes y etiquetas de un dataset de MNIST,
    y las devuelve como arrays de numpy.
    """
    return torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()



