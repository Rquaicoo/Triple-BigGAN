from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Input, Conv2DTranspose, Reshape, Activation, Flatten, Embedding, Concatenate, AveragePooling2D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

def classifier():
  base_model = ResNet50(
      input_shape=(256, 256, 3),
      include_top = False,
      weights = 'imagenet'
  )

  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  number_of_classes = 10

  # multilayer perceptron with 2 residual blocks and skip connections
  shortcut = x


  x = Dense(512, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  x = Concatenate()([x, shortcut])

  shortcut = x

  x = Dense(512, activation='relu')(x)
  x = Dense(2048, activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  x = Concatenate()([x, shortcut])

  preds = Dense(number_of_classes, activation="softmax")(x)

  #create model for classifier

  model = Model(inputs=base_model.input, outputs=preds)

  return model

