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

# 0. Hyper-parameters
epochs = 1 # Change to larger number for better accuracy (longer training time)
batch_size = 128 # Change to a smaller number if you don't have enough memory
show_figures = 3 # Visualize ES results at the end of the training process
visualize_first_sample = False


# 1. Download MNIST Dataset
train_dataset = datasets.MNIST('.', train=True, download=True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST('.', train=False, download=True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Length of the train dataset:", len(train_dataset))
print("Length of the test dataset:", len(test_dataset))


# 2. Visualize a sample
if visualize_first_sample:
    index = 1234 # <-- change to another number to see another sample
    image = train_dataset[index][0] # torch.Tensor([1,28,28]): Number of channels = 1 (Gray scale image), Image dimensions = 28x28 (pixels)
    label = train_dataset[index][1] # torch.Tensor([1]): The corresponding label

    print('Input sample #{}:'.format(index))
    plt.imshow(image.squeeze(), 'gray')
    plt.show()
    print('The correct label for the image:', label)


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


# 4. Train the network with back-propagation
model = Net() # Initialize the network above
optimizer = torch.optim.AdamW(model.parameters()) # A utility function to update network weights: w[t+1] = w[t] + learning rate * gradient_of_w
criterion = nn.CrossEntropyLoss() # This is a differentiable objective function: \sum{y_true[i] * log(y_pred[i])}

# This keeps a record of accuracy scores and train time for each iteration
acc_list_grad = []
time_list_grad = [0]

for epoch in range(epochs): # Number of train iterations
    print('Iteration #{}'.format(epoch))
    start_t = time.time() # Start time counter
    loss = train(train_loader, model, optimizer, criterion) # Train the network with the train dataset
    end_t = time.time() # Stop time counter
    time_list_grad.append(time_list_grad[-1] + end_t - start_t) # Save the train duration

    loss, outputs, targets = test(test_loader, model, criterion) # Test the network with the test dataset
    acc = metrics.accuracy_score(targets, outputs.argmax(-1)) # Compute accuracy score
    print('Test accuracy score:', acc)
    if len(acc_list_grad) == 0 or acc > max(acc_list_grad): # Save the model if we get better test accuracy score
        save('MNIST_Grad.pt', model)
        print('Save model to MNIST_Grad.pt')
    acc_list_grad.append(acc) # Save the test accuracy score


# 5. Define training function for ES
def distribute_task(params):
    model, theta, source, target = params
    params_to_model(theta, model)
    output = model(source)
    return criterion(output, target).item()

def evolve(mu, sigma, npop, data_loader, model, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()
    running_loss = 0
    
    normal = Normal(device)
    expectation = Expectation()

    # This part is for multiprocessing computation, however it doesn't show any significant improvement in training time.
    # ----------- 
    # import copy
    # import multiprocess as mp
    # pool = mp.Pool(2) # Change to the highest number of CPUs
    # models = [] # Create N copies of the model to avoid accessing the sharing parameters (one process can overwrite data of another process, very dangerous!!!)
    # for j in range(npop):
    #     models.append(copy.deepcopy(model))
    # ----------- 
    
    prog_bar = tqdm(data_loader)
    for i, (source, target) in enumerate(prog_bar):
        source = source.to(device)
        target = target.to(device)

        theta, ratio = normal(mu, sigma, npop)
        score = torch.zeros_like(ratio)

        # This part is for multiprocessing computation only, it works but no training time improvement.
        # ----------- 
        # params = []
        # for j in range(npop):
        #     params.append((models[j],theta[j],source,target))
        # score = torch.tensor(pool.map(distribute_task, params), device=device)
        # ----------- 
        
        # This part sequentially compute score (no parallel)
        # ----------- 
        for j in range(npop):
            params_to_model(theta[j], model)
            output = model(source)
            score[j] = criterion(output, target).item()
        # ----------- 
        
        loss = expectation(normalize(score), ratio)
        running_loss += score.mean()
        prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))

        # Back-propagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

    return running_loss / len(data_loader)


# 6. Train the network with ES
model = Net()
# Turn off gradient computation (this will make the network cannot back-propagate, thus becomes a black-box optimization)
for param in model.parameters():
    param.requires_grad = False

flat_param = torch.cat(model_to_params(model), dim=0) # From the network, take out and flatten parameters into an array.
mu = torch.tensor(flat_param, requires_grad=True, device='cpu') # Initialize mean for the parameters
sigma = torch.full(mu.shape, 0.01, requires_grad=True, device='cpu') # Initialize standard deviations for the parameters
npop = 100 # How many individual in the population
optimizer = torch.optim.AdamW([mu,sigma]) # A utility function to update network weights: w[t+1] = w[t] + learning rate * gradient_of_w
criterion = nn.CrossEntropyLoss() # Although this objective function is differentiable, we turned off the gradient computation in the above step, hence it becomes a non-differentiable objective function.

# This keeps a record of accuracy scores and train time for each iteration
acc_list_es = []
time_list_es = [0]

for epoch in range(epochs): # Number of train iterations
    print('Iteration #{}'.format(epoch))
    start_t = time.time() # Start time counter
    loss = evolve(mu, sigma, npop, train_loader, model, optimizer, criterion, device=mu.device) # Evolve the network with the train dataset
    end_t = time.time() # Stop time counter
    time_list_es.append(time_list_es[-1] + end_t - start_t) # Save the train duration

    params_to_model(mu, model) # Put the parameters back into the network for testing
    loss, outputs, targets = test(test_loader, model, criterion) # Test the network with the test dataset
    acc = metrics.accuracy_score(targets, outputs.argmax(-1)) # Compute accuracy score
    print('Test accuracy score:', acc)
    if len(acc_list_es) == 0 or acc > max(acc_list_es): # Save the model if we get better test accuracy score
        save('MNIST_ES.pt', model)
        print('Save model to MNIST_ES.pt')
    acc_list_es.append(acc) # Save the test accuracy score


# 7. Visualize the performance of both methods
import numpy as np
plt.plot(np.array(time_list_grad)[1:], np.array(acc_list_grad), 'g', label='Gradient-based Method')
plt.plot(np.array(time_list_es)[1:], np.array(acc_list_es), 'r', label='Evolutionary Strategy')
plt.legend()
plt.xlabel('Time (second)')
plt.ylabel('Accuracy')
plt.show()

# 8. Visualize the outputs
import random
for i in range(show_figures): 
    index = random.randint(0,len(outputs)) # <-- change to another number to see another sample
    image = test_dataset[index][0] # torch.Tensor([1,28,28]): Number of channels = 1 (Gray scale image), Image dimensions = 28x28 (pixels)
    label = test_dataset[index][1] # torch.Tensor([1]): The corresponding label
    output = outputs[index].argmax(-1).item() 
    print('Input sample #{}:'.format(index))
    plt.imshow(image.squeeze(), 'gray')
    plt.show()
    print('The predicted label for the image:', output)
    print('The correct label for the image:', label)
plt.close()
