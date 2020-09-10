import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras import callbacks
import tensorflow.compat.v2 as tf

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