#! /usr/bin/env python

import logging

import tensorflow as tf
import tifffile as tiff

from PIL import Image as img
from astropy.io import fits

from starnet_v1_TF2 import StarNet


def main():
    starnet = StarNet(mode='Greyscale')
    starnet.load_model('./weights', './history')

    #in_name = "./rgb_test5.tif"
    #out_name = "../../Nemesis/starnet/test.tif"
    in_name = "../../Nemesis/starnet/Orion_B_IRAC_Field1_i1.resamp.stretched.fits"
    out_name = "../../Nemesis/starnet/Orion_B_IRAC_Field1_i1.resamp.starless.fits"

    starnet.transform(in_name, out_name)


if __name__ == "__main__":
    main()
    exit(0)
