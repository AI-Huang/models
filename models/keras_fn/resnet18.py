#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-01-21 23:19
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/applications/resnet.py
# @RefLink : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# @RefLink : https://github.com/keras-team/keras-applications/blob/bc89834ed3/keras_applications/resnet_common.py

"""resnet18 based on tf.keras.applications.resnet
# Reference:
    - [Deep Residual Learning for Image Recognition](
        https://arxiv.org/abs/1512.03385) (CVPR 2015)
# Tested environment:
    tensorflow==2.1.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for TensorFlow r2.4
# from tensorflow.python.keras.applications.resnet import block1, stack1, ResNet
from keras_applications.resnet_common import block1, stack1, ResNet

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


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
    """Instantiates the ResNetxx architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 2, stride1=1, name='conv2')
        x = stack1(x, 128, 2, name='conv3')
        x = stack1(x, 256, 2, name='conv4')
        return stack1(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet18', include_top, weights,
                  input_tensor, input_shape, pooling, classes, **kwargs)


def main_test():
    model_type = "ResNet18"
    input_shape = (224, 224, 3)
    num_classes = 2
    model = ResNet18(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=num_classes
    )

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=model_type + ".png", show_shapes=True)


def main():
    main_test()


if __name__ == "__main__":
    main()
