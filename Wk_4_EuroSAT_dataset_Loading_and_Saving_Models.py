# create a neural network that classifies land uses and land covers from satellite imagery
# Save model using Tensorflow's callbacks and reload it later.
# Compare results with a pre-trained neural network classifier

'''https://github.com/phelber/EuroSAT
EuroSAT dataset: 27000 labelled Sentinel-2 satellite images of different land uses:
residential, industrial, highway, river, forest,
pasture, herbaceous vegetation, annual crop, permanent crop and sea/lake.

For a reference, see the following papers:

Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber,
Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and
Remote Sensing, 2019.

Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick
Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018. '''

# Goal is to construct a neural network that classifies a satellite image into one of these 10 classes,
# as well as applying some of the saving and loading techniques you have learned in the
# previous sessions.

import tensorflow as tf
# had to install TFDS separately as a stand alone package pip install tensorflow_datasets
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''Data: Train model on a subset of the total data, with 20,000 images, roughly 2000 images per class (split into 
4000 test images and 16000 training images) '''


def load_eurosat_data():
    data_dir = 'data/'
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_eurosat_data()

# Note when we created the data set we already divided the images by 255 so we don't need to do that here

tmp = 1251
img = x_train[tmp]
plt.imshow(img)
plt.show()

print('shape of single image:', x_train[0].shape)

# create grid of 3x3 images to visualize the data set
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    img = x_train[i]
    plt.imshow(img)

plt.show()


# Set up 6 layer model using Sequential API
def get_new_model(shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(shape), padding='SAME', name='conv_1'),
        Conv2D(8, (3, 3), activation='relu', padding='SAME', name='conv_2'),
        MaxPooling2D((8, 8), name='pool_1'),
        Flatten(name='flatten'),
        Dense(32, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    return model


model = get_new_model(x_train[0].shape)


# Evaluate a model's test accuracy

def get_test_accuracy(model, x_test, y_test):
    """Test model classification accuracy"""
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    # print('accuracy: {acc:0.3f}'.format(acc=test_acc))
    return test_loss, test_acc


# Print the model summary and calculate its initialised test accuracy
model.summary()

opt = tf.keras.optimizers.Adam()
acc = tf.keras.metrics.Accuracy()


def compile_model(model):
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

compile_model(model)

# only 10% accuracy initially without training (since it performs at chance levels)
test_loss, test_acc = get_test_accuracy(model, x_test, y_test)
"Accuracy before training (chance levels): {:5.2f}%".format(100 * test_acc)


# Create checkpoints to save model during training.
# Create three callbacks:
# checkpoint_every_epoch: checkpoint that saves the model weights every epoch during training
# checkpoint_best_only: checkpoint that saves only the weights with the highest validation accuracy.
# Use the testing data as the validation data.
# early_stopping:early stopping object that ends training if the validation accuracy has not improved in 3 epochs.
def get_callbacks():
    checkpoint_path = 'checkpoints_every_epoch/checkpoint-{epoch:04d}.ckpt'
    checkpoint_every_epoch = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, frequency='epoch')
    checkpoint_best_only = ModelCheckpoint(filepath='checkpoints_best_only',
                                           save_weights_only=True,  # if false will save entire model
                                           save_freq='epoch',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, verbose=1)
    return checkpoint_every_epoch, checkpoint_best_only, early_stopping


callbacks = get_callbacks()

# train the model with the new callbacks
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks,
                    verbose=0)

df = pd.DataFrame(history.history)
df.plot(y=['accuracy', 'val_accuracy'])
plt.show()

latest = tf.train.latest_checkpoint('checkpoints_every_epoch')
print(latest)


# create a new model with the saved weights
def get_model_last_epoch(model):
    model.load_weights(latest)
    compile_model(model)
    # re-evaluate the model
    last_epoch_test_loss, last_epoch_test_acc = get_test_accuracy(model, x_test, y_test)
    print("Model with last epoch weights, Accuracy: {:5.2f}%".format(100 * last_epoch_test_acc))
    model_last_epoch = model
    return model_last_epoch, last_epoch_test_loss, last_epoch_test_acc


def get_model_best_epoch(model):
    model.load_weights('checkpoints_best_only')
    compile_model(model)
    # re-evaluate the model
    best_epoch_test_loss, best_epoch_test_acc = get_test_accuracy(model, x_test, y_test)
    print("Model with best epoch weights, Accuracy: {:5.2f}%".format(100 * best_epoch_test_acc))
    model_best_epoch = model
    return model_best_epoch, best_epoch_test_loss, best_epoch_test_acc



#Model with last epoch weights, Accuracy: 81.88%
#Model with best epoch weights, Accuracy: 82.17%
model_last_epoch = get_model_last_epoch(get_new_model(x_train[0].shape))
model_best_epoch = get_model_best_epoch(get_new_model(x_train[0].shape))

####################################
#Read in a pretrained model for the EuroSat data and compare the results to my model
'''This is how I would load a pretrained model, however
the train/test data I used is formatted differently than 
the data used to create this model (y has 10 categories in my 
dataset instead of 1). So I have a shape mismatch error, I'd have
to redownload the data in the correct format to make this work, 
but since my models work well I'm not going to download it
def get_model_eurosatnet():
    model=load_model('EuroSatNet.h5')
    return model

# Print a summary of the EuroSatNet model, along with its validation accuracy.

model_eurosatnet = get_model_eurosatnet()
model_eurosatnet.summary()
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)

#test_loss, test_acc=get_test_accuracy(model_eurosatnet, x_test, y_test)
#print("Model with best epoch weights, Accuracy: {:5.2f}%".format(100 * test_acc))
'''