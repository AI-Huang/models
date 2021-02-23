#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-01-21 23:19
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/applications/resnet.py
# @RefLink : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
# @RefLink : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# @RefLink : https://github.com/keras-team/keras-applications/blob/bc89834ed3/keras_applications/resnet_common.py

"""resnet18 based on tf.keras.applications.resnet
# Reference:
    - [Deep Residual Learning for Image Recognition](
        https://arxiv.org/abs/1512.03385) (CVPR 2015)
# Tested environment:
    tensorflow==2.1.0
    tensorflow==2.3.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pkg_resources import parse_version
if parse_version(tf.__version__) > parse_version("2.1"):
    # for TensorFlow r2.2 and above
    from tensorflow.python.keras.applications.resnet import block1, stack1, ResNet

    def keras_modules_injection(base_fun):
        # Do nothing
        def wrapper(*args, **kwargs):
            return base_fun(*args, **kwargs)
        return wrapper
else:
    # for TensorFlow r2.1 and below
    from keras_applications.resnet_common import block1, stack1, ResNet
    from tensorflow.python.keras.applications import keras_modules_injection

from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.util.tf_export import keras_export


def basicblock1(x, filters=64, kernel_size=3, stride=1,
                conv_shortcut=True, name=None, **kwargs):
    """A basic residual block.
    From PyTorch: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L37

    BasicBlock for ResNet18 and ResNet34 according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.

    BasicBlock only has TWO convolution layers.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    expansion: int = 1
    if conv_shortcut is True:
        shortcut = layers.Conv2D(expansion * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def basicstack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = basicblock1(x, filters, stride=stride1, name=name + '_basicblock1')
    for i in range(2, blocks + 1):
        x = basicblock1(x, filters, conv_shortcut=False,
                        name=name + '_basicblock' + str(i))
    return x


@keras_export('keras.applications.resnet.ResNet18',
              'keras.applications.ResNet18')
@keras_modules_injection
def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = basicstack1(x, 64, 2, stride1=1, name='conv2')
        x = basicstack1(x, 128, 2, name='conv3')
        x = basicstack1(x, 256, 2, name='conv4')
        return basicstack1(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet18', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.resnet.ResNet18Bottleneck',
              'keras.applications.ResNet18Bottleneck')
@keras_modules_injection
def ResNet18Bottleneck(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       classes=1000,
                       **kwargs):
    """Instantiates the ResNet18Bottleneck architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 2, stride1=1, name='conv2')
        x = stack1(x, 128, 2, name='conv3')
        x = stack1(x, 256, 2, name='conv4')
        return stack1(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet18_bottleneck', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.resnet.ResNet34',
              'keras.applications.ResNet34')
@keras_modules_injection
def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = basicstack1(x, 64, 3, stride1=1, name='conv2')
        x = basicstack1(x, 128, 4, name='conv3')
        x = basicstack1(x, 256, 6, name='conv4')
        return basicstack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet34', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


@keras_export('keras.applications.resnet.ResNet18Bottleneck',
              'keras.applications.ResNet18Bottleneck')
@keras_modules_injection
def ResNet34Bottleneck(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       classes=1000,
                       **kwargs):
    """Instantiates the ResNet34Bottleneck architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 2, stride1=1, name='conv2')
        x = stack1(x, 128, 2, name='conv3')
        x = stack1(x, 256, 2, name='conv4')
        return stack1(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet34_bottleneck', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)
