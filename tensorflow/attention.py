#!usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os, time
import cv2
import argparse, random
import skimage.transform
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scipy.io

tf.compat.v1.disable_eager_execution()

print(tf.__version__)

print(tf.test.is_gpu_available())

# Configuration
IMG_SIZE = 112
EPOCHS = 20
BATCH_SIZE = 256
BUFFER_SIZE = 1000

delta = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
delta_i = delta[-1]

PATH = os.path.abspath('.') + "/dataset/PlacePulse"
IMG_PATH = PATH + "/images/2011/"
LABEL_PATH = PATH + "/labels/"

SAVE_PATH = 'saved_models/models/'

num_test_sample = 50

units = 256
embedding_dim = 512

input_shape = (112,112,1)
output_shape = 10

mat_data_train = scipy.io.loadmat('dataset/MNIST_mod/MNIST_data_train_re.mat')
mat_data_test = scipy.io.loadmat('dataset/MNIST_mod/MNIST_data_test_re.mat')

train_x = np.float32(mat_data_train['X_train'])
train_y = np.float32(mat_data_train['Y_train'])

test_x = np.float32(mat_data_test['X_test'])#[:9900, :]
test_y = np.float32(mat_data_test['Y_test'])#[:9900, :]

print(train_x.shape, train_y.shape)
train_x = np.expand_dims(train_x, axis=3)
test_x = np.expand_dims(test_x, axis=3)
print(train_x.shape, train_y.shape)

# validation_x = mat_data_test['X_test'][9900:, :]
# validation_y = mat_data_test['Y_test'][9900:, :]

del mat_data_train
del mat_data_test

print("Train data shape: " + str(train_x.shape))
print("Train label shape: " + str(train_y.shape))
print("Test data shape: " + str(test_x.shape))
print("Test label shape: " + str(test_y.shape))
# print("Validation data shape: " + str(validation_x.shape))
# print("Validation label shape: " + str(validation_y.shape))
print(np.max(test_x), np.min(test_x))

def reset_state(batch_size, units):
  return tf.zeros((batch_size, units))
    
def loss_function(real, pred):
  #pred_ = tf.nn.softmax(pred)
  #loss_ = cost(real, pred_)
  cost = tf.nn.sigmoid_cross_entropy_with_logits(labels = real, logits = pred)
  return tf.reduce_mean(cost)
    
class SoftAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, units, name='AttentionLayer', **kwargs):
    self.units = units
    super(SoftAttentionLayer, self).__init__(name=name, **kwargs)

    #super(SoftAttentionLayer, self).__init__(**kwargs)
    #self.name = name
  def build(self, input_shape):
    self.W1 = tf.keras.layers.Dense(self.units, kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())
    self.W2 = tf.keras.layers.Dense(self.units, kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())
    self.V = tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())
    
  def call(self, features, hidden):

    # features(CNN_encoder output) shape == (batch_size, 196, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 196, hidden_size)
    #print("feature:", features.shape)
    #print("hidden:", hidden_with_time_axis.shape)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 196, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)
    #attention_weights = tf.reshape(attention_weights, (-1, ))

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    #print("weights", attention_weights.shape, "feat", features.shape, "context", context_vector.shape)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    #print("context:", context_vector.shape)

    return attention_weights, context_vector  #alpha, z

  def get_config(self):
    config = {'units': self.units,
             }
    base_config = super(SoftAttentionLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    
#if not os.path.exists(checkpoint_file):
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, LSTMCell, Reshape, Dense, GlobalAveragePooling2D

input_image = Input(shape=input_shape, name='input_image')
hidden = Input(shape=(units), name="hidden_states") # h
memmory = Input(shape=(units), name="memmory_states") # c

h = reset_state(1, units)
c = reset_state(1, units)

x = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),padding='same',use_bias=True, activation="relu", name="conv_1", kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())(input_image)
x = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2),padding='same',use_bias=True, activation="relu", name="conv_2", kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())(x)
x = Conv2D(filters=embedding_dim, kernel_size=(3,3), strides=(2,2),padding='same',use_bias=True, activation="relu", name="conv_3", kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())(x)

x = Reshape((-1, embedding_dim))(x)

#alpha, z = SoftAttentionLayer(units, name="att_1")(x, hidden)
#lstm_input = Concatenate(axis=1)([z, hidden])
#_, [h, c] = LSTMCell(units)(lstm_input, [hidden, memmory])

alpha, z = SoftAttentionLayer(units, name="att_1")(x, h)
lstm_input = Concatenate(axis=1)([z, h])
_, [h, c] = LSTMCell(units)(lstm_input, [h, c])

alpha, z = SoftAttentionLayer(units, name="att_2")(x, h)
lstm_input = Concatenate(axis=1)([z, h])
_, [h, c] = LSTMCell(units)(lstm_input, [h, c])

alpha, z = SoftAttentionLayer(units, name="att_3")(x, h)
lstm_input = Concatenate(axis=1)([z, h])
_, [h, c] = LSTMCell(units)(lstm_input, [h, c])

alpha, z = SoftAttentionLayer(units, name="att_4")(x, h)
lstm_input = Concatenate(axis=1)([z, h])
_, [h, c] = LSTMCell(units)(lstm_input, [h, c])

y = Dense(output_shape, activation="linear", name="predictions", kernel_initializer=tf.initializers.GlorotUniform(), bias_initializer=tf.initializers.GlorotUniform())(h)

#att_model = Model(inputs=[input_image, hidden, memmory], outputs=[y])
att_model = Model(inputs=[input_image], outputs=[y])
att_model.summary()

print("alpha", alpha.shape,"z", z.shape, "h",hidden.shape, "c", memmory.shape,"x", x.shape)

tf.keras.utils.plot_model(att_model, show_shapes=True, to_file=SAVE_PATH+'att_model.png')

optm = tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-8)
#optimizer = tf.keras.optimizers.RMSprop()
#loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.CategoricalCrossentropy()

att_model.compile(optimizer=optm, loss=loss_function, metrics=['accuracy', 'mse'])

train_h = np.zeros((train_x.shape[0], units))
train_c = np.zeros((train_x.shape[0], units))
print(train_x.shape, train_h.shape, train_c.shape)

#history = att_model.fit([train_x, train_h, train_c], train_y, batch_size= BATCH_SIZE, epochs= EPOCHS, validation_split=0.2, verbose=1, shuffle=True, callbacks = [checkpoints])

from tensorflow.keras.models import load_model

model_to_predict=att_model
#model_to_predict = load_model(checkpoint_file, custom_objects={'SoftAttentionLayer':SoftAttentionLayer, "loss_function":loss_function})

test_h = np.zeros((test_x.shape[0], units))
test_c = np.zeros((test_x.shape[0], units))

#evaluations = model_to_predict.evaluate([test_x, test_h, test_c], test_y, batch_size=BATCH_SIZE, verbose=1)
evaluations = model_to_predict.evaluate(test_x, test_y, batch_size=1, verbose=1)
metric_loss = model_to_predict.metrics_names[0]
loss_value = evaluations[0]
metric_acc = model_to_predict.metrics_names[1]
acc_value = evaluations[1]
metric_mse = model_to_predict.metrics_names[2]
mse_value = evaluations[2]
print("%s: %.2f%%" % (metric_loss, loss_value*100))
print("%s: %.2f%%" % (metric_acc, acc_value*100))
print("%s: %.2f%%" % (metric_mse, mse_value*100))

from vis.visualization import visualize_cam
from vis.utils import utils

layer_idx = utils.find_layer_idx(model_to_predict, 'predictions')
penultimate_layer_idx = utils.find_layer_idx(model_to_predict, "conv_3")

seed_input = test_x[0]
print(seed_input.shape)

y_pred            = model_to_predict.predict(seed_input[np.newaxis,...])
class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
class_idx  = class_idxs_sorted[0]

viscam_img = visualize_cam(model_to_predict, layer_idx, class_idx, seed_input,
                           penultimate_layer_idx = penultimate_layer_idx, 
                           backprop_modifier = None, grad_modifier = None)

def plot_map(_img, grads, class_idx, y_pred):
  fig, axes = plt.subplots(1,2,figsize=(14,5))
  axes[0].imshow(_img, cmap="gray")
  axes[1].imshow(_img, cmap="gray")
  i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
  fig.colorbar(i)
  plt.suptitle("Pr(class={}) = {:5.2f}".format(class_idx, y_pred[0,class_idx]))
  plt.savefig(SAVE_PATH+"sample_cam.png", bbox_inches='tight', pad_inches = 0)
  
  plt.clf()
  plt.cla()
  plt.close()
 
plot_map(seed_input, viscam_img, class_idx, y_pred)
