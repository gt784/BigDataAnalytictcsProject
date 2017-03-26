import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random

#configuration and hyperparameters
# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['dogs', 'cats']
num_classes = len(classes)

# batch size
batch_size = 16

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'data/train_data'
test_path = 'data/test_data'
checkpoint_dir = "C:/Projects/playground/tensorflow/tf_image_clf/models/"

#load the data

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)