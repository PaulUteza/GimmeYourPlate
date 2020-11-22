#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow.compat.v1 as tf
from pylab import rcParams


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def preprocess(to_predict):
    to_predict = np.array(to_predict)/255
    if len(to_predict.shape) == 2:
        to_predict = to_predict[..., np.newaxis]
    to_predict = tf.image.resize_image_with_pad(to_predict, 64, 128, align_corners=False)
    if to_predict.shape[2] == 1:
        to_predict = to_predict[:,:,0]
    return to_predict


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Create placeholders for image data and expected point positions

class Model(object):
    xxx = 0


# Build neural network
def build_model(PIXEL_COUNT, LABEL_COUNT):
    x_placeholder = tf.placeholder(tf.float32, shape=[None,PIXEL_COUNT])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])

    x_image = tf.reshape(x_placeholder, [-1, 64, 128, 1])
    # Convolution Layer 1
    W_conv1 = weight_variable("w1", [3, 3, 1, 32])
    b_conv1 = bias_variable("b1", [32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Convolution Layer 2
    W_conv2 = weight_variable("w2", [2, 2, 32, 64])
    b_conv2 = bias_variable("b2", [64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Convolution Layer 3
    W_conv3 = weight_variable("w3", [2, 2, 64, 128])
    b_conv3 = bias_variable("b3", [128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # Dense layer 1
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*16*128])
    W_fc1 = weight_variable("w4", [8*16*128, 500])
    b_fc1 = bias_variable("b4", [500])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    # Dense layer 2
    W_fc2 = weight_variable("w5", [500, 500])
    b_fc2 = bias_variable("b5", [500])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # Output layer
    W_out = weight_variable("w6", [500, LABEL_COUNT])
    b_out = bias_variable("b6", [LABEL_COUNT])

    output = tf.matmul(h_fc2, W_out) + b_out
    model = Model()
    model.x_placeholder = x_placeholder
    model.y_placeholder = y_placeholder
    model.output = output

    return model


def show_image(image, labels):
    rect = Rectangle((labels[0], labels[1]), labels[2]-labels[0], labels[3]-labels[1], edgecolor='r', fill=False)
    plt.imshow(image)
    gca = plt.gca()
    gca.add_patch(rect)
    return rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()


def plot_images(images, labels):
    rcParams['figure.figsize'] = 14, 8
    plt.gray()
    fig = plt.figure()
    plt.axis('off')
    labels = np.mean(labels,axis=0)
    x, y, width, height = show_image(images, labels)

    return fig, x, y, width, height


def result(to_predict, x, y, height, width):
    crop_img = to_predict[int(y):int(y)+ int(height), int(x):int(x)+int(width)]
    return crop_img


def make_prediction(img, MODEL_PATH):
    to_predict = preprocess(img)
    x_img = to_predict.shape[0]
    y_img = to_predict.shape[1]
    if len(to_predict.shape) == 3:
        z_img = to_predict.shape[2]
    else:
        z_img = 1
    PIXEL_COUNT = x_img * y_img * z_img
    LABEL_COUNT = 1
    to_predict_reshape = np.reshape(to_predict, (1, x_img * y_img * z_img))
    g = tf.Graph()
    with g.as_default():
        session = tf.InteractiveSession()
        model = build_model(PIXEL_COUNT, LABEL_COUNT)
        saver = tf.train.Saver()
        saver.restore(session, MODEL_PATH)
        predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: to_predict_reshape})
        session.close()
    box_image, x, y, width, height = plot_images(to_predict, (predictions+1) * (64, 32, 64, 32))
    plate = result(to_predict, x, y, height, width)
    return box_image, plate

