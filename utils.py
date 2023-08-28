from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Input, Conv2DTranspose, Reshape, Activation, Flatten, Embedding, Concatenate, AveragePooling2D, Embedding
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

weight_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)


def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2], keepdims=True)
    return gsp

def max_pooling(x) :
    x = MaxPooling2D(pool_size=2, strides=2, padding='SAME')(x)
    return x

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.compat.v1.get_variable(name + 'u', [1, w_shape[-1]],
                      initializer=tf.keras.initializers.TruncatedNormal(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar

def conv(name, inputs, nums_out, k_size, strides, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.name_scope(name):
        W = tf.compat.v1.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.keras.initializers.Orthogonal())
        b = tf.compat.v1.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
        con = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
    return tf.nn.bias_add(con, b)


def inner_product(global_pooled, y, nums_class, update_collection=None):
    W = global_pooled.shape[-1]
    V = tf.compat.v1.get_variable("V", [nums_class, W], initializer=tf.compat.v1.orthogonal_initializer())
    V = tf.transpose(V)
    V = spectral_normalization("embed", V, update_collection=update_collection)
    V = tf.transpose(V)
    temp = tf.nn.embedding_lookup(V, y)
    temp = tf.reduce_sum(temp * global_pooled, axis=1, keepdims=True)
    return temp

def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.name_scope(name):
        W = tf.compat.v1.get_variable("W", [nums_in, nums_out], initializer=tf.compat.v1.orthogonal_initializer())
        b = tf.compat.v1.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, W), b)
    return temp

def non_local(name, inputs, update_collection, is_sn):
    h, w, num_channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    location_num = h * w
    downsampled_num = location_num // 4
    with tf.name_scope(name):
        theta = conv("f", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        theta = tf.reshape(theta, [-1, location_num, num_channels // 8])
        phi = conv("h", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        phi = AveragePooling2D()(phi)
        phi = tf.reshape(phi, [-1, downsampled_num, num_channels // 8])
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        g = conv("g", inputs, num_channels // 2, 1, 1, update_collection, is_sn)
        g = AveragePooling2D()(g)
        g = tf.reshape(g, [-1, downsampled_num, num_channels // 2])
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
        sigma = tf.compat.v1.get_variable("sigma_ratio", [], initializer=tf.constant_initializer(0.0))
        attn_g = conv("attn", attn_g, num_channels, 1, 1, update_collection, is_sn)
        return inputs + sigma * attn_g


