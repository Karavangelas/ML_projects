import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras import callbacks
import tensorflow.compat.v2 as tf

# load and save dataset
dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

# split to train, validation and test set, test is not used in the training process
train_set = tfds.load("oxford_flowers102", split='train', as_supervised=True)
validation_set = tfds.load("oxford_flowers102", split='validation', as_supervised=True)
test_set = tfds.load("oxford_flowers102", split='test', as_supervised=True)

# preprocess and data augmentation
def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize_with_pad(cropped_image, 224, 224, antialias=True)
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# apply processing and augmentation to train and validation sets
batch_size = 32
train_set = train_set.shuffle(1000).repeat()
train_set = train_set.map(partial(preprocess, randomize=False)).batch(batch_size).prefetch(1)
validation_set = validation_set.map(partial(preprocess, randomize=False)).batch(batch_size).prefetch(1)

# create base model, add AveragePooling layer and one Dense layer
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# freeze all Xception layers to train the first time
for layer in base_model.layers:
    layer.trainable = False
    
# first time train, adding early stop and save callbacks
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
save = keras.callbacks.ModelCheckpoint("flowersModel.h5", monitor='val_loss', save_best_only=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
    metrics=["accuracy"])
history = model.fit(train_set,
    steps_per_epoch=int(0.75 * dataset_size / batch_size),
    validation_data=validation_set,
    validation_steps=int(0.5* dataset_size / batch_size),
    epochs=20, callbacks = [early_stop, save], verbose=2)

# unfreeze all previous layers for second time train

for layer in base_model.layers:
    layer.trainable = True
    
# train a second time

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
save = keras.callbacks.ModelCheckpoint("flowersModel.h5", monitor='val_loss', save_best_only=False)
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=validation_set,
                    validation_steps=int(0.5 * dataset_size / batch_size),
                    epochs=20, callbacks = [early_stop, save], verbose=2)
