from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
from ..parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy as copy_module
import re
import math
from scipy import misc, ndimage
from skimage import transform as tf
import itertools
import cv2
import six
import six.moves as sm
from . import Augmenter, Add

# brightness is just an alias for Add
AdjustBrightness = Add

class AdjustContrast(Augmenter):
    """Adjusts the contrast of RGB or grayscale images

    For each channel, this Op computes the mean of the image pixels in the channel and then adjusts
    each component x of each pixel to (x - mean) * contrast_factor + mean.

    (copies tf.image.adjust_contrast())

    Parameters
    ----------
    factor : int or (int, int)
        the factor with which to modify the contrast. +ve = more, -ve = less.
        If in tuple form, the amount used for each pixel is drawn from a uniform rand
        If StochasticParameter, left as-is

    name : string, optional(default=None)
        name of the instance

    deterministic : boolean, optional (default=False)
        Whether random state will be saved before augmenting images
        and then will be reset to the saved value post augmentation
        use this parameter to obtain transformations in the EXACT order
        everytime

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, factor, name=None, deterministic=False, random_state=None):
        super(AdjustContrast, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_float(factor):
            assert factor >= 0, "contrast factor must be above 0, was %d" % factor
            self.factor = Deterministic(factor)
        elif ia.is_iterable:
            assert len(factor) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(factor),)
            for f in factor:
                assert f >= 0, "contrast range must be above 0, was %d" % f
            self.factor = Uniform(factor[0], factor[1])
        elif isinstance(factor, StochasticParameter):
            self.factor = factor
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(factor),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        # (x - mean) * contrast_factor + mean
        for i, image in enumerate(images):
            image = image.astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            contrast_factor = self.factor.draw_sample(random_state=random_state)
            mean = np.mean(image)
            result[i] = np.clip(((image - mean) * contrast_factor + mean), 0, 255).astype(np.uint8)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.factor]
