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
from . import Augmenter

def per_image_standardization(images):
    """Linearly scales image to have zero mean and unit norm.

    This op computes (x - mean) / adjusted_stddev, where mean is the average of all values in
    image, and adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements())).

    stddev is the standard deviation of all values in image. It is capped away from zero to
    protect against division by 0 when handling uniform images.

    (copies `tf.image.per_image_standardization()` but using numpy instead of tensors)

    Parameters
    ----------
    images : numpy array in the shape (batch_size, height, width, channels)
        The images to linearly standardize
    """
    return np.array(map(
        lambda image:
            ((image.astype(np.float64) - np.mean(image)) /
            max(np.std(images), 1.0 / np.sqrt(image.size))),
        images))

class ContrastNormalization(Augmenter):
    """Augmenter class for ContrastNormalization

    Parameters
    ----------
    alpha : float, iterable of len 2, StochasticParameter
        Normalization parameter that governs the contrast ratio
        of the resulting image

    per_channel : boolean, optional(default=False)
        apply transform in a per channel manner

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
    def __init__(self, alpha=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(ContrastNormalization, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            assert alpha >= 0.0, "Expected alpha to have range (0, inf), got value %.4f." % (alpha,)
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel:
                nb_channels = images[i].shape[2]
                alphas = self.alpha.draw_samples((nb_channels,), random_state=rs_image)
                for c, alpha in enumerate(alphas):
                    image[..., c] = alpha * (image[..., c] - 128) + 128
            else:
                alpha = self.alpha.draw_sample(random_state=rs_image)
                image = alpha * (image - 128) + 128
            np.clip(image, 0, 255, out=image)
            result[i] = image.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha]


