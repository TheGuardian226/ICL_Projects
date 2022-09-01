import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

def scale_mnist_data(train_images, test_images):
    train_images = train_images / 255.
    test_images = test_images / 255.
    
    return train_images, test_images

# scale the data
scaled_train_images, scaled_test_images = scale_mnist_data(train_images, test_images)


scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

def get_model(input_shape):
    model = tf.keras.Sequential([
        Conv2D(8, (3,3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
    
model = get_model(scaled_train_images[0].shape)

# compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

compile_model(model)

# fit the model
def train_model(model, scaled_train_images, train_labels):
    history = model.fit(scaled_train_images, train_labels, epochs=5)
    return history

history = train_model(model, scaled_train_images, train_labels)

# evaluate the model
def evaluate_model(model, scaled_test_images, test_labels):
    test_loss, test_acc = model.evaluate(scaled_test_images, test_labels)
    return test_loss, test_acc

test_loss, test_acc = evaluate_model(model, scaled_test_images, test_labels)
print(test_loss, test_acc)
