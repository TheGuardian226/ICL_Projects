import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np

def load_eurosat_data():
    data_dir = 'data/'
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_test  = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_eurosat_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create a model
def get_new_model(input_shape):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape, padding='same', name='conv_1'),
        Conv2D(8, (3,3), activation='relu', padding='same', name='conv_2'),
        MaxPooling2D((8,8), name='pool_1'),
        Flatten(name='flatten'),
        Dense(32, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = get_new_model(x_train[0].shape)

def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))

# Callback that saves weights only every epoch, into directory named "checkpoints_every_epoch", with filname "checkpoint_XXX" containing epoch number formatted to three digits
def get_checkpoint_every_epoch():
    get_checkpoint_every_epoch = ModelCheckpoint(filepath='checkpoints_every_epoch/checkpoint_{epoch:03d}', save_weights_only=True)
    return get_checkpoint_every_epoch

# Callback that saves weights that generates the highest validation accuracy, saving into directory named "checkpoints_best_only", with filename "checkpoint"
def get_checkpoint_best_only():
    get_checkpoint_best_only = ModelCheckpoint(filepath='checkpoints_best_only/checkpoint', save_weights_only=True, monitor='val_accuracy', save_best_only=True)
    return get_checkpoint_best_only

# Callback that stops training when validation accuracy has not improved for 3 epochs
def get_early_stopping():
    get_early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
    return get_early_stopping


# Create callbacks

checkpoint_every_epoch = get_checkpoint_every_epoch()
checkpoint_best_only = get_checkpoint_best_only()
early_stopping = get_early_stopping()

# Train model using the callbacks
callbacks = [checkpoint_every_epoch, checkpoint_best_only, early_stopping]
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=callbacks)

# Load the latest saved epoch
def get_model_last_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('checkpoints_every_epoch'))
    return model

# Load the best saved epoch
def get_model_best_epoch(model):
    model.load_weights(tf.train.latest_checkpoint('checkpoints_best_only'))
    return model

