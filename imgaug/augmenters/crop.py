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

class Crop(Augmenter):
    """Crop Augmenter object that crops input image(s)

    Parameters
    ----------
    window : int or tuple, optional(default=None)
        randomly choose the position with a fixed window.
        single int -> window by window images
        tuple of ints -> window[0] by window[1] images
        images in (height, width) format)

    center : boolean, optional(default=False)
        whether to crop from the center only when using the 'window' mode

    px : int or tuple, optional(default=None)
        crop a constant, or in a provided range amount of pixels from each the
        (top, right, bottom, left) in that order.

    percent : tuple, optional(default=None)
        percent crop on each of the axis

    keep_size : boolean, optional(default=True)
        whether to interpolate the result back to the original image's size

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
    def __init__(self, window=None, center=False, px=None, percent=None, keep_size=True, name=None,
                 deterministic=False, random_state=None):
        super(Crop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.keep_size = keep_size

        if window is not None:
            self.mode = "window"
            self.center = center
            if ia.is_single_integer(window):
                self.window = (window, window)
            elif type(window) is tuple and len(window) == 2:
                self.window = window
            return

        self.all_sides = None
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None
        if px is None and percent is None:
            self.mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only crop by pixels or percent, not both.")
        elif px is not None:
            self.mode = "px"
            if ia.is_single_integer(px):
                assert px >= 0
                #self.top = self.right = self.bottom = self.left = Deterministic(px)
                self.all_sides = Deterministic(px)
            elif isinstance(px, tuple):
                assert len(px) in [2, 4]
                def handle_param(p):
                    if ia.is_single_integer(p):
                        assert p >= 0
                        return Deterministic(p)
                    elif isinstance(p, tuple):
                        assert len(p) == 2
                        assert ia.is_single_integer(p[0])
                        assert ia.is_single_integer(p[1])
                        assert p[0] >= 0
                        assert p[1] >= 0
                        return DiscreteUniform(p[0], p[1])
                    elif isinstance(p, list):
                        assert len(p) > 0
                        assert all([ia.is_single_integer(val) for val in p])
                        assert all([val >= 0 for val in p])
                        return Choice(p)
                    elif isinstance(p, StochasticParameter):
                        return p
                    else:
                        raise Exception("Expected int, tuple of two ints, list of ints or StochasticParameter, got type %s." % (type(p),))

                if len(px) == 2:
                    #self.top = self.right = self.bottom = self.left = handle_param(px)
                    self.all_sides = handle_param(px)
                else: # len == 4
                    self.top = handle_param(px[0])
                    self.right = handle_param(px[1])
                    self.bottom = handle_param(px[2])
                    self.left = handle_param(px[3])
            elif isinstance(px, StochasticParameter):
                self.top = self.right = self.bottom = self.left = px
            else:
                raise Exception("Expected int, tuple of 4 ints/lists/StochasticParameters or StochasticParameter, git type %s." % (type(px),))
        else: # = elif percent is not None:
            self.mode = "percent"
            if ia.is_single_number(percent):
                assert 0 <= percent < 1.0
                #self.top = self.right = self.bottom = self.left = Deterministic(percent)
                self.all_sides = Deterministic(percent)
            elif isinstance(percent, tuple):
                assert len(percent) in [2, 4]
                def handle_param(p):
                    if ia.is_single_number(p):
                        return Deterministic(p)
                    elif isinstance(p, tuple):
                        assert len(p) == 2
                        assert ia.is_single_number(p[0])
                        assert ia.is_single_number(p[1])
                        assert 0 <= p[0] < 1.0
                        assert 0 <= p[1] < 1.0
                        return Uniform(p[0], p[1])
                    elif isinstance(p, list):
                        assert len(p) > 0
                        assert all([ia.is_single_number(val) for val in p])
                        assert all([0 <= val < 1.0 for val in p])
                        return Choice(p)
                    elif isinstance(p, StochasticParameter):
                        return p
                    else:
                        raise Exception("Expected int, tuple of two ints, list of ints or StochasticParameter, got type %s." % (type(p),))

                if len(percent) == 2:
                    #self.top = self.right = self.bottom = self.left = handle_param(percent)
                    self.all_sides = handle_param(percent)
                else: # len == 4
                    self.top = handle_param(percent[0])
                    self.right = handle_param(percent[1])
                    self.bottom = handle_param(percent[2])
                    self.left = handle_param(percent[3])
            elif isinstance(percent, StochasticParameter):
                self.top = self.right = self.bottom = self.left = percent
            else:
                raise Exception("Expected number, tuple of 4 numbers/lists/StochasticParameters or StochasticParameter, got type %s." % (type(percent),))


    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            height, width = images[i].shape[0:2]
            if self.mode == "window":
                if self.center:
                    top = images[i].shape[0] / 2 - self.window[0] / 2
                    left = images[i].shape[1] / 2 - self.window[1] / 2
                else:
                    top = np.random.randint(images[i].shape[0] - self.window[0])
                    left = np.random.randint(images[i].shape[1] - self.window[1])

                bottom = height - top - self.window[0]
                right = width - left - self.window[1]
            else:
                top, right, bottom, left = self._draw_samples_image(seed, height, width)
            image_cropped = images[i][top:height-bottom, left:width-right, :]
            if self.keep_size:
                image_cropped = ia.imresize_single_image(image_cropped, (height, width))
            result.append(image_cropped)

        if not isinstance(images, list):
            if self.keep_size:
                result = np.array(result, dtype=np.uint8)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            seed = seeds[i]
            height, width = keypoints_on_image.shape[0:2]
            top, right, bottom, left = self._draw_samples_image(seed, height, width)
            shifted = keypoints_on_image.shift(x=-left, y=-top)
            shifted.shape = (height - top - bottom, width - left - right)
            if self.keep_size:
                result.append(shifted.on(keypoints_on_image.shape))
            else:
                result.append(shifted)

        return result

    def _draw_samples_image(self, seed, height, width):
        random_state = ia.new_random_state(seed)

        if self.all_sides is not None:
            samples = self.all_sides.draw_samples((4,), random_state=random_state)
            top, right, bottom, left = samples
        else:
            rs_top = random_state
            rs_right = rs_top
            rs_bottom = rs_top
            rs_left = rs_top
            top = self.top.draw_sample(random_state=rs_top)
            right = self.right.draw_sample(random_state=rs_right)
            bottom = self.bottom.draw_sample(random_state=rs_bottom)
            left = self.left.draw_sample(random_state=rs_left)

        if self.mode == "px":
            # no change necessary for pixel values
            pass
        elif self.mode == "percent":
            # percentage values have to be transformed to pixel values
            top = int(height * top)
            right = int(width * right)
            bottom = int(height * bottom)
            left = int(width * left)
        else:
            raise Exception("Invalid mode")

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)
        if remaining_height < 1:
            regain = abs(remaining_height) + 1
            regain_top = regain // 2
            regain_bottom = regain // 2
            if regain_top + regain_bottom < regain:
                regain_top += 1

            if regain_top > top:
                diff = regain_top - top
                regain_top = top
                regain_bottom += diff
            elif regain_bottom > bottom:
                diff = regain_bottom - bottom
                regain_bottom = bottom
                regain_top += diff

            assert regain_top <= top
            assert regain_bottom <= bottom

            top = top - regain_top
            bottom = bottom - regain_bottom

        if remaining_width < 1:
            regain = abs(remaining_width) + 1
            regain_right = regain // 2
            regain_left = regain // 2
            if regain_right + regain_left < regain:
                regain_right += 1

            if regain_right > right:
                diff = regain_right - right
                regain_right = right
                regain_left += diff
            elif regain_left > left:
                diff = regain_left - left
                regain_left = left
                regain_right += diff

            assert regain_right <= right
            assert regain_left <= left

            right = right - regain_right
            left = left - regain_left

        assert top >= 0 and right >= 0 and bottom >= 0 and left >= 0
        assert top + bottom < height
        assert right + left < width

        return top, right, bottom, left

    def get_parameters(self):
        return [self.top, self.right, self.bottom, self.left]


