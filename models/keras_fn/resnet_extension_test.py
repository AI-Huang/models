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
from tensorflow.keras.utils import plot_model


def main_test():
    input_shape = (224, 224, 3)
    num_classes = 2

    from models.keras_fn.resnet_extension import ResNet18, ResNet34, ResNet18Bottleneck, ResNet34Bottleneck
    model_names = ["ResNet18", "ResNet34",
                   "ResNet18Bottleneck", "ResNet34Bottleneck"]
    test_nets = [ResNet18, ResNet34, ResNet18Bottleneck, ResNet34Bottleneck]

    test_official = True
    if test_official:
        from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
        model_names += ["ResNet50", "ResNet101", "ResNet152"]
        test_nets += [ResNet50, ResNet101, ResNet152]

    for i, model_name in enumerate(model_names):
        net = test_nets[i]
        model = net(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )
        plot_model(model, to_file=f"./png/{model_name}.png", show_shapes=True)
        print(f"Saved model png to './png/{model_name}.png'")


def main():
    main_test()


if __name__ == "__main__":
    main()
