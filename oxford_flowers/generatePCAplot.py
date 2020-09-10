import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras import callbacks
import tensorflow.compat.v2 as tf
import numpy as np
from sklearn.decomposition import PCA

# load and save dataset
dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

# load test_set
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

# load pretrained model without top layer and add average pooling layer
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
output = keras.layers.GlobalAveragePooling2D()(base_model.output)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# stack test set on top of it and predict feature vectors
evalset,info = tfds.load(name='oxford_flowers102', split='test',as_supervised=True,with_info=True)
evalPipe=evalset.map(preprocess,num_parallel_calls=16).batch(128).prefetch(1)
for feats,lab in evalPipe.unbatch().batch(6000).take(1):
    probPreds=model.predict(feats)
    
# generate and save graph
pca = PCA(random_state=1)
pca.fit(probPreds)
var = pca.explained_variance_ratio_

plt.plot((np.cumsum(var)))
plt.grid(True)
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.savefig("explainedVariancePlot.png")