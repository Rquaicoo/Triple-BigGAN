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
from .utils import non_local
from SpectralNormalization import SpectralNormalization

def generator_residual_block(input_tensor, filters):
  shortcut = input_tensor

  x = BatchNormalization()(input_tensor)
  x = ReLU()(x)
  x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

  shortcut = Conv2D(filters, kernel_size=(3, 3), padding='same')(shortcut)

  #add shortcut to 2nd convolutional layer
  x = shortcut + x

  return x



def generator_network():
  z_dim = 128
  channel_width_multiplier = 128
  number_of_classes = 10

  # z: input noise vector, y: embedding
  z_input, y_input = Input(shape=(z_dim,)), Input(shape=(number_of_classes,))

  input_concat = Concatenate()([z_input, y_input])
  #generator layers
  #TODO: use a loop to clean up code

  ch = channel_width_multiplier * 16

  x = Dense(4 * 4 * ch, activation='relu')(input_concat)
  print("shape", x.shape)
  x = tf.keras.layers.Reshape((4, 4, ch))(x)

  print(x.shape)
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)
  ch = ch // 2
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)
  ch = ch // 2
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)
  ch = ch // 2
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)

  #TODO: Add Non-Local Block
  x = non_local("Non-Local", x, None, True)

  ch = ch // 2
  x = generator_residual_block(x, filters=ch)
  x = SpectralNormalization(Conv2DTranspose(ch, kernel_size=(3,3), strides=(2,2), padding='same'))(x)

  print(x.shape)

  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    
  print(x.shape)  

  model = tf.keras.models.Model(inputs=[z_input, y_input], outputs=x)

  return model

