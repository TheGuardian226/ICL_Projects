import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms # import datasets and transforms from torchvision
import torch.nn.functional as F # import the functional module from torch.nn

mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True) # load the data in batches of 100
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

W1 = torch.randn(784, 500)/np.sqrt(784)
W1.requires_grad_(True)
B1 = torch.zeros(500, requires_grad=True)

W2 = torch.randn(500, 10)/np.sqrt(500)
W2.requires_grad_(True)
B2 = torch.zeros(10, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W1, B1, W2, B2], lr=0.1)


# Iterate 
for images, labels in train_loader:
    optimizer.zero_grad() # reset the gradients to 0
    
    # Forward pass
    x1 = images.view(-1, 28*28) # flatten the images into a vector
    y1 = torch.matmul(x1, W1) + B1 # y1 is the output of the first layer
    
    x2 = F.relu(y1) # apply the ReLU activation function
    y2 = torch.matmul(x2, W2) + B2 # y2 is the output of the second layer
    #Compute the loss over both layers
    cross_entropy = F.cross_entropy(y2, labels)
    # Backward pass
    cross_entropy.backward() # compute the gradients of W and b with respect to the loss
    optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate over the test data
    for images, labels in test_loader:
        x1 = images.view(-1, 28*28)
        y1 = torch.matmul(x1, W1) + B1
        x2 = F.relu(y1)
        y2 = torch.matmul(x2, W2) + B2
        
        predictions = torch.argmax(y2, dim=1)
        correct += torch.sum((predictions == labels).float())

print("Accuracy: {}".format(correct/total))
