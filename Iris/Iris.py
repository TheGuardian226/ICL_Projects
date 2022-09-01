from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def read_in_and_split_data(iris_data):
    # Split the data into train and test
    train_data, test_data, train_targets, test_targets = model_selection.train_test_split(iris_data.data, iris_data.target, test_size=0.1, random_state=42)
    # Return the train and test data
    return train_data, test_data, train_targets, test_targets

iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)

# Convert targets to a one-hot encoding
train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))

# Create the model
def get_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=input_shape, kernel_initializer='HeUniform', bias_initializer='ones', activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

model = get_model(train_data[0].shape)

# Compile the model
def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
compile_model(model)

# Fit the model
def train_model(model, train_data, train_targets, epochs):
    history = model.fit(train_data, train_targets, epochs=epochs, batch_size=40, validation_split=0.15)
    return history

history = train_model(model, train_data, train_targets, epochs=800)

# Define better model
def get_regularised_model(input_shape, dropout_rate, weight_decay):
    model = Sequential([
        Dense(64, input_shape=input_shape, kernel_initializer=tf.keras.intializers.he_uniform(), bias_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), activation='relu'),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        Dense(3, activation='softmax')
    ])
    return model

reg_model = get_regularised_model(train_data[0].shape, 0.3, 0.001)

compile_model(reg_model)

reg_history = train_model(reg_model, train_data, train_targets, epochs=800)

# Create callback to introduce Early Stopping and Reduce Learning Rate on Plateau
def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
    learning_rate_reduction = ReduceLROnPlateau(factor=0.2, patience=20)
    return [early_stopping, learning_rate_reduction]

call_model = get_regularised_model(train_data[0].shape, 0.3, 0.0001)
compile_model(call_model)
early_stopping, learning_rate_reduction = get_callbacks()
call_history = call_model.fit(train_data, train_targets, epochs=800, validation_split=0.15,
                         callbacks=[early_stopping, learning_rate_reduction], verbose=0)
