from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Input, Conv2DTranspose, Reshape, Activation, Flatten, Embedding, Concatenate, AveragePooling2D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from keras.layers.pooling.average_pooling2d import AveragePooling2D
from utils import non_local, global_sum_pooling

import time
import numpy as np
import matplotlib.pyplot as plt

def discriminator_residual_block(input_tensor, filters):
  shortcut = input_tensor

  x = ReLU()(input_tensor)

  x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

  x = ReLU()(x)

  x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

  #add shortcut to 2nd convolutional layer

  shortcut = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

  return x + shortcut



def discriminator_network():
  input_shape = (256, 256, 3)
  channel_width_multiplier = 128
  num_classes = 10

  # Input layer
  input_image = Input(shape=input_shape)

  x = discriminator_residual_block(input_image, 3)  # ResBlock down: 3 -> ch
  print(x.shape)

  ch = channel_width_multiplier

  x = discriminator_residual_block(x, ch)  # ResBlock down: ch -> ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  #TODO: Add Non-Local Block              Non-Local Block
  x = non_local("Non-Local", x, None, True)

  x = discriminator_residual_block(x, ch)  # ResBlock down: ch -> 2 . ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  ch*=2
  x = discriminator_residual_block(x, ch)  # ResBlock down: 2 . ch -> 4 . ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  ch*=2
  x = discriminator_residual_block(x, ch)  # ResBlock down: 4 . ch -> 8 . ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  ch *=2
  x = discriminator_residual_block(x, ch)  # ResBlock down: 8 . ch -> 16 . ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  ch *=2
  x = discriminator_residual_block(x, ch)  # ResBlock down: 16 . ch -> 16 . ch
  x = Conv2D(ch, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
  print(x.shape)

  x = ReLU()(x)
  x = global_sum_pooling(x)
  print(x.shape)

  class_labels = Input(shape=(10,))
  h = Dense(1)(class_labels)

  x = Flatten()(x)

  x = Concatenate()([x, h])

  output = Dense(1)(x)
  print(output.shape)


  model = Model(inputs=[input_image, class_labels], outputs=output)

  return model