#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-02-21 20:19
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/applications/applications_test.py

"""Integration tests for resnet18.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from models.keras_fn.resnet18 import ResNet18


def main_test():
    model_name = "ResNet18"
    input_shape = (224, 224, 3)
    num_classes = 2
    model = ResNet18(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=num_classes
    )

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=model_name + ".png", show_shapes=True)


def main():
    main_test()


if __name__ == "__main__":
    main()
