# -*- coding: utf-8 -*-
# Author: Nguyen, Tran Nhat Hoang
# Project: Classifying hand-written digits using Evolutionary Strategy
# Open-source packages to download:
# pip install numpy
# pip install torch
# pip install torchvision
# pip install matplotlib
# pip install scikit-learn
# pip install tqdm

import torch
import torch.nn as nn # To build a neural network
import torchvision.datasets as datasets # To load the MNIST dataset
import torchvision.transforms as transforms # To convert PIL object to a tensor
import matplotlib.pyplot as plt # To visualize the dataset
from utils import train, test, save, load # My implemented utility functions
from es import model_to_params, params_to_model, normalize, Normal, Expectation # My implementation of ES, fully compatible with PyTorch library
from tqdm import tqdm # To show progress bar
import sklearn.metrics as metrics # To compute accuracy score
import time # To measure how long to run a training iteration

# 1. Download MNIST Dataset
train_dataset = datasets.MNIST('.', train=True, download=True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST('.', train=False, download=True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Length of the train dataset:", len(train_dataset))
print("Length of the test dataset:", len(test_dataset))

# 3. Build a neural network to classify hand-written digits
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,8), nn.ReLU(), # Input to hidden layer #1
            nn.Linear(8,8), nn.ReLU(), # Hidden layer #1 to hidden layer #2
            nn.Linear(8,10), # Hidden layer #2 to output layer
        )

    def forward(self, input):
        input = input.view(input.shape[0], -1) # Flatten the image (128x1x28x28 tensor) into (128x784 pixels)
        return self.model(input)
    
# 4. Load the saved model
saved_model = 'MNIST_ES.pt'
model = Net() # Initialize the network
print('Reload {} into model'.format(saved_model))
load(saved_model, model)
loss, outputs, targets = test(test_loader, model) # Test the network with the test dataset
acc = metrics.accuracy_score(targets, outputs.argmax(-1)) # Compute accuracy score
print('Test accuracy score:', acc)

saved_model = 'MNIST_Grad.pt'
model = Net() # Initialize the network
print('Reload {} into model'.format(saved_model))
load(saved_model, model)
loss, outputs, targets = test(test_loader, model) # Test the network with the test dataset
acc = metrics.accuracy_score(targets, outputs.argmax(-1)) # Compute accuracy score
print('Test accuracy score:', acc)
