import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms # Import datasets and transforms from torchvision
import torch.nn.functional as F # Import the functional module from torch.nn

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
    optimizer.zero_grad() # Reset the gradients to 0
    
    # Forward pass
    x1 = images.view(-1, 28*28) # Flatten the images into a vector
    y1 = torch.matmul(x1, W1) + B1 # y1 is the output of the first layer
    
    x2 = F.relu(y1) # Apply the ReLU activation function
    y2 = torch.matmul(x2, W2) + B2 # y2 is the output of the second layer
    # Compute the loss over both layers
    cross_entropy = F.cross_entropy(y2, labels)
    # Backward pass
    cross_entropy.backward() # Compute the gradients of W and b with respect to the loss
    optimizer.step()

# Model will take in input files and output the predicted label
import torch
from PIL import Image
import torchvision.transforms as transforms

exit = False

while not exit:
    print("1. Predict")
    print("2. Exit")
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        image_name = input("Enter the name of the image you want to classify with extension included: ")
        
        # Read a PIL image
        try:
            image = Image.open(image_name).convert('L')
        except:
            print("Error: Image not found")
            continue
        image1 = image.resize((28, 28)) # Resize the image to 28x28
        # Define a transform to convert PIL 
        # image to a Torch tensor
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        
        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        img_tensor = transform(image1)

        tensor = img_tensor.view(-1, 28*28)

        tensor = tensor/255.0

        layer_1_output = torch.matmul(tensor, W1) + B1 
        layer_2_input = F.relu(layer_1_output) # Apply the ReLU activation function
        layer_2_output = torch.matmul(layer_2_input, W2) + B2 
        predictions = torch.argmax(layer_2_output, dim=1) # Get the index of the highest value in each row

        print("The predicted number is:", predictions) # Print the predictions
        
        continue
    
    elif choice == 2:
        exit = True
        print("Exiting...")
