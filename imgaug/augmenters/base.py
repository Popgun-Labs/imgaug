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

@six.add_metaclass(ABCMeta)
class Augmenter(object):
    """Base class for Augmenter objects
    Parameters
    ----------
    name : string, optional
        Name given to an Augmenter object
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

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(Augmenter, self).__init__()

        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        self.deterministic = deterministic

        if random_state is None:
            if self.deterministic:
                self.random_state = ia.new_random_state()
            else:
                self.random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.activated = True

    def augment_batches(self, batches, hooks=None):
        """Augment images, batch-wise
        Parameters
        ----------
        batches : array-like, shape = (num_samples, height, width, channels)
            image batch to augment
        hooks : optional(default=None)
            HooksImages object to dynamically interfere with the Augmentation process
        Returns
        -------
        augmented_batch : array-like, shape = (num_samples, height, width, channels)
            corresponding batch of augmented images
        """
        assert isinstance(batches, list)
        return [self.augment_images(batch, hooks=hooks) for batch in batches]

    def augment_image(self, image, hooks=None):
        """Augment a single image
        Parameters
        ----------
        image : array-like, shape = (height, width, channels)
            The image to augment
        hooks : optional(default=None)
            HooksImages object to dynamically interfere with the Augmentation process
        Returns
        -------
        img : array-like, shape = (height, width, channels)
            The corresponding augmented image
        """
        assert len(image.shape) == 3, "Expected image to have shape (height, width, channels), got shape %s." % (image.shape,)
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """Augment multiple images
        Parameters
        ----------
        images : array-like, shape = (num_samples, height, width, channels) or
                 a list of images (particularly useful for images of various
                 dimensions)
            images to augment
        parents : optional(default=None)
            # TODO
        hooks : optional(default=None)
            HooksImages object to dynamically interfere with the Augmentation process
        Returns
        -------
        images_result : array-like, shape = (num_samples, height, width, channels)
            corresponding augmented images
        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksImages()

        if ia.is_np_array(images):
            assert len(images.shape) == 4, "Expected 4d array of form (N, height, width, channels), got shape %s." % (str(images.shape),)
            assert images.dtype == np.uint8, "Expected dtype uint8 (with value range 0 to 255), got dtype %s." % (str(images.dtype),)
            images_tf = images
        elif ia.is_iterable(images):
            if len(images) > 0:
                assert all([len(image.shape) == 3 for image in images]), "Expected list of images with each image having shape (height, width, channels), got shapes %s." % ([image.shape for image in images],)
                assert all([image.dtype == np.uint8 for image in images]), "Expected dtype uint8 (with value range 0 to 255), got dtypes %s." % ([str(image.dtype) for image in images],)
            images_tf = list(images)
        else:
            raise Exception("Expected list/tuple of numpy arrays or one numpy array, got %s." % (type(images),))

        if isinstance(images_tf, list):
            images_copy = [np.copy(image) for image in images]
        else:
            images_copy = np.copy(images)

        images_copy = hooks.preprocess(images_copy, augmenter=self, parents=parents)

        if hooks.is_activated(images_copy, augmenter=self, parents=parents, default=self.activated):
            if len(images) > 0:
                images_result = self._augment_images(
                    images_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                self.random_state.uniform()
            else:
                images_result = images_copy
        else:
            images_result = images_copy

        images_result = hooks.postprocess(images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        if isinstance(images_result, list):
            assert all([image.dtype == np.uint8 for image in images_result]), "Expected list of dtype uint8 as augmenter result, got %s." % ([image.dtype for image in images_result],)
        else:
            assert images_result.dtype == np.uint8, "Expected dtype uint8 as augmenter result, got %s." % (images_result.dtype,)

        return images_result

    @abstractmethod
    def _augment_images(self, images, random_state, parents, hooks):
        raise NotImplementedError()

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """Augment image keypoints
        Parameters
        ----------
        keypoints_on_images : # TODO
        parents : optional(default=None)
            # TODO
        hooks : optional(default=None)
            HooksImages object to dynamically interfere with the Augmentation process
        Returns
        -------
        keypoints_on_images_result : # TODO
        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksKeypoints()

        assert ia.is_iterable(keypoints_on_images)
        assert all([isinstance(keypoints_on_image, ia.KeypointsOnImage) for keypoints_on_image in keypoints_on_images])

        keypoints_on_images_copy = [keypoints_on_image.deepcopy() for keypoints_on_image in keypoints_on_images]

        keypoints_on_images_copy = hooks.preprocess(keypoints_on_images_copy, augmenter=self, parents=parents)

        if hooks.is_activated(keypoints_on_images_copy, augmenter=self, parents=parents, default=self.activated):
            if len(keypoints_on_images_copy) > 0:
                keypoints_on_images_result = self._augment_keypoints(
                    keypoints_on_images_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                self.random_state.uniform()
            else:
                keypoints_on_images_result = keypoints_on_images_copy
        else:
            keypoints_on_images_result = keypoints_on_images_copy

        keypoints_on_images_result = hooks.postprocess(keypoints_on_images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        return keypoints_on_images_result

    @abstractmethod
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError()

    # TODO most of the code of this function could be replaced with ia.draw_grid()
    def draw_grid(self, images, rows, cols):
        if ia.is_np_array(images):
            if len(images.shape) == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [images[:, :, np.newaxis]]
            else:
                raise Exception("Unexpected images shape, expected 2-, 3- or 4-dimensional array, got shape %s." % (images.shape,))
        assert isinstance(images, list)

        det = self if self.deterministic else self.to_deterministic()
        augs = []
        for image in images:
            augs.append(det.augment_images([image] * (rows * cols)))

        augs_flat = list(itertools.chain(*augs))
        cell_height = max([image.shape[0] for image in images] + [image.shape[0] for image in augs_flat])
        cell_width = max([image.shape[1] for image in images] + [image.shape[1] for image in augs_flat])
        width = cell_width * cols
        height = cell_height * (rows * len(images))
        grid = np.zeros((height, width, 3))
        for row_idx in range(rows):
            for img_idx, image in enumerate(images):
                for col_idx in range(cols):
                    image_aug = augs[img_idx][(row_idx * cols) + col_idx]
                    cell_y1 = cell_height * (row_idx * len(images) + img_idx)
                    cell_y2 = cell_y1 + image_aug.shape[0]
                    cell_x1 = cell_width * col_idx
                    cell_x2 = cell_x1 + image_aug.shape[1]
                    grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image_aug

        return grid

    def show_grid(self, images, rows, cols):
        """Quickly show examples results of the applied augmentation
        Parameters
        ----------
        images : array-like, shape = (num_samples, height, width, channels) or
                 a list of images (particularly useful for images of various
                 dimensions)
            images to augment
        rows : integer.
            number of rows in the grid
        cols : integer
            number of columns in the grid
        """
        grid = self.draw_grid(images, rows, cols)
        misc.imshow(grid)

    def to_deterministic(self, n=None):
        """ # TODO
        """
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """ # TODO
        """
        aug = self.copy()
        aug.random_state = ia.new_random_state()
        aug.deterministic = True
        return aug

    def reseed(self, deterministic_too=False, random_state=None):
        """For reseeding the internal random_state
        Parameters
        ----------
        deterministic_too : boolean, optional(default=False)
            # TODO
        random_state : np.random.RandomState instance, optional(default=None)
            random seed generator
        """
        if random_state is None:
            random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            pass # just use the provided random state without change
        else:
            random_state = ia.new_random_state(random_state)

        if not self.deterministic or deterministic_too:
            seed = random_state.randint(0, 10**6, 1)[0]
            self.random_state = ia.new_random_state(seed)

        for lst in self.get_children_lists():
            for aug in lst:
                aug.reseed(deterministic_too=deterministic_too, random_state=random_state)

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    def get_children_lists(self):
        return []

    def find_augmenters(self, func, parents=None, flat=True):
        """ # TODO
        """
        if parents is None:
            parents = []

        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug.find_augmenters(func, parents=subparents, flat=flat)
                if len(found) > 0:
                    if flat:
                        result.extend(found)
                    else:
                        result.append(found)
        return result

    def find_augmenters_by_name(self, name, regex=False, flat=True):
        """Find augmenter(s) by name
        Parameters
        ----------
        name : string
            name of the augmenter to find
        regex : regular Expression, optional(default=False)
            Regular Expression for searching the augmenter
        flat : boolean, optional(default=True)
            # TODO
        Returns
        -------
        found augmenter instance
        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """Find augmenters by names
        Parameters
        ----------
        names : list of strings
            names of the augmenter to find
        regex : regular Expression, optional(default=False)
            Regular Expression for searching the augmenter
        flat : boolean, optional(default=True)
            # TODO
        Returns
        -------
        found augmenter instance(s)
        """
        if regex:
            def comparer(aug, parents):
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        else:
            return self.find_augmenters(lambda aug, parents: aug.name in names, flat=flat)

    def remove_augmenters(self, func, copy=True, noop_if_topmost=True):
        """Remove Augmenters from the list of augmenters
        Parameters
        ----------
        func : # TODO
        copy : boolean, optional(default=True)
            removing the augmenter or it's copy
        noop_if_topmost : boolean, optional(default=True)
            if func is provided and noop_if_topmost is True
            an object of Noop class is returned
        Returns
        -------
        aug : removed augmenter object
        """
        if func(self, []):
            if not copy:
                raise Exception("Inplace removal of topmost augmenter requested, which is currently not possible.")

            if noop_if_topmost:
                return Noop()
            else:
                return None
        else:
            aug = self if not copy else self.deepcopy()
            aug.remove_augmenters_inplace(func, parents=[])
            return aug

    def remove_augmenters_inplace(self, func, parents):
        """Inplace removal of augmenters
        Parameters
        ----------
        func : # TODO
        parents : # TODO
        """
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, aug) in enumerate(to_remove):
                # self._remove_augmenters_inplace_from_list(lst, aug, i, i - count_removed)
                del lst[i - count_removed]

            for aug in lst:
                aug.remove_augmenters_inplace(func, subparents)

    # TODO
    # def to_json(self):
    #    pass

    def copy(self):
        """Obtain a copy"""
        return copy_module.copy(self)

    def deepcopy(self):
        """Obtain a deep copy"""
        return copy_module.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (self.__class__.__name__, self.name, params_str, self.deterministic)

class Noop(Augmenter):
    """Noop is an Augmenter that does nothing

    Parameters
    ----------
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
    def __init__(self, name=None, deterministic=False, random_state=None):
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(Noop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []
