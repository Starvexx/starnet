#! /usr/bin/env python

import logging

import tensorflow as tf
import tifffile as tiff

from PIL import Image as img
from astropy.io import fits

from starnet_v1_TF2 import StarNet


def main():
    ### Config GPU to prevent NotFoundError: No algorithm worked! when using Conv2D
    ### See issue #43174 on github:
    ### https://github.com/tensorflow/tensorflow/issues/43174

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1280)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    ###
    ###
    ###

    starnet = StarNet(mode='Greyscale', stride=128)
    starnet.load_model('./weights', './history')

    #in_name = "./rgb_test5.tif"
    #out_name = "../../Nemesis/starnet/test.tif"
    in_name = "../../test/test.fits"
    out_name = "../../test/starles.fits"

    starnet.transform(in_name, out_name)


if __name__ == "__main__":
    main()
    exit(0)
