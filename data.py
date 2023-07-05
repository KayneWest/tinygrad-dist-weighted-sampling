# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import random

import numpy as np

#import mxnet as mx
#from mxnet import nd

import cv2
from tinygrad.tensor import Tensor

def split_data(data, num_slice, batch_axis=0, even_split=True):
    """Splits an NDArray into `num_slice` slices along `batch_axis`.
    Usually used for data parallelism where each slices is sent
    to one device (i.e. GPU).

    Parameters
    ----------
    data : NDArray
        A batch of data.
    num_slice : int
        Number of desired slices.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.
        If `True`, an error will be raised when `num_slice` does not evenly
        divide `data.shape[batch_axis]`.

    Returns
    -------
    list of NDArray
        Return value is a list even if `num_slice` is 1.
    """
    size = data.shape[batch_axis]
    if even_split and size % num_slice != 0:
        raise ValueError(
            f"data with shape {str(data.shape)} cannot be evenly split into {num_slice} slices " \
            f"along axis {batch_axis}. Use a batch size that's multiple of {num_slice} " \
            f"or set even_split=False to allow uneven partitioning of data.")

    n_each_section, extras = divmod(size, num_slice)
    section_sizes = [0] + (extras * [n_each_section + 1] +
                           (num_slice - extras) * [n_each_section])
    div_points = np.array(section_sizes).cumsum()
    if is_np_array():
        slices = _mx_np.split(data, indices_or_sections=list(div_points[1: -1]), axis=batch_axis)
    else:
        slices = []
        for i in range(num_slice):
            st = div_points[i]
            end = div_points[i + 1]
            slices.append(ndarray.slice_axis(data, axis=batch_axis, begin=st, end=end))
    return slices


def split_and_load(data, ctx_list, batch_axis=0, even_split=True):
    """Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.

    Parameters
    ----------
    data : NDArray or ndarray
        A batch of data.
    ctx_list : list of Context
        A list of Contexts.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArrays or ndarrays
        Each corresponds to a context in `ctx_list`.
    """
    array_fn = _mx_np.array if is_np_array() else ndarray.array
    if not isinstance(data, ndarray.NDArray):
        data = array_fn(data, ctx=ctx_list[0])
    if len(ctx_list) == 1:
        return [data.as_in_context(ctx_list[0])]

    slices = split_data(data, len(ctx_list), batch_axis, even_split)
    return [i.as_in_context(ctx) for i, ctx in zip(slices, ctx_list)]


def _get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError(f'Unknown interp method {interp}')
    return interp

def scale_down(src_size, size):
    """Scales down crop size if it's larger than image size.

    If width/height of the crop is larger than the width/height of the image,
    sets the width/height to the width/height of the image.

    Parameters
    ----------
    src_size : tuple of int
        Size of the image in (width, height) format.
    size : tuple of int
        Size of the crop in (width, height) format.

    Returns
    -------
    tuple of int
        A tuple containing the scaled crop size in (width, height) format.

    Example
    --------
    >>> src_size = (640,480)
    >>> size = (720,120)
    >>> new_size = mx.img.scale_down(src_size, size)
    >>> new_size
    (640,106)
    """
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)

def random_crop(src, size, interp=2):
    """Randomly crop `src` with `size` (width, height).
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    src: Source image `NDArray`
    size: Size of the crop formatted as (width, height). If the `size` is larger
           than the image, then the source image is upsampled to `size` and returned.
    interp: int, optional, default=2
        Interpolation method. See resize_short for details.
    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    Tuple
        A tuple (x, y, width, height) where (x, y) is top-left position of the crop in the
        original image and (width, height) are the dimensions of the cropped image.

    Example
    -------
    >>> im = mx.nd.array(cv2.imread("flower.jpg"))
    >>> cropped_im, rect  = mx.image.random_crop(im, (100, 100))
    >>> print cropped_im
    <NDArray 100x100x1 @cpu(0)>
    >>> print rect
    (20, 21, 100, 100)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size.

    Parameters
    ----------
    src : NDArray
        Input image
    x0 : int
        Left boundary of the cropping area
    y0 : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : tuple of (w, h)
        Optional, resize to new size after cropping
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    """
    out = src[y0:y0+h, x0:x0+w]
    if size is not None and (w, h) != size:
        sizes = (h, w, size[1], size[0])
        out = cv2.resize(out, *size, interp=_get_interp_method(interp, sizes))
    return out

def center_crop(src, size, interp=2):
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is smaller
    than `size`.

    .. note:: This requires MXNet to be compiled with USE_OPENCV.

    Parameters
    ----------
    src : NDArray
        Binary source image data.
    size : list or tuple of int
        The desired output image size.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        The cropped image.
    Tuple
        (x, y, width, height) where x, y are the positions of the crop in the
        original image and width, height the dimensions of the crop.

    Example
    -------
    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.image.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> cropped_image, (x, y, width, height) = mx.image.center_crop(image, (1000, 500))
    >>> cropped_image
    <NDArray 500x1000x3 @cpu(0)>
    >>> x, y, width, height
    (1241, 910, 1000, 500)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def transform(data, target_wd, target_ht, is_train, box):
    """Crop and normnalize an image nd array."""
    if box is not None:
        x, y, w, h = box
        data = data[y:min(y+h, data.shape[0]), x:min(x+w, data.shape[1])]

    # Resize to target_wd * target_ht.
    data = cv2.resize(data, (target_wd, target_ht))

    # Normalize in the same way as the pre-trained model.
    data = data.astype(np.float32) / 255.0
    # data /= 255.0
    data = (data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    if is_train:
        if random.random() < 0.5:
            data = np.flip(data, axis=1)
        data, _ = random_crop(data, (224, 224))
    else:
        data, _ = center_crop(data, (224, 224))

    # Transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht).
    data = np.transpose(data, (2, 0, 1))

    # If image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = np.tile(data, (3, 1, 1))
    return data.reshape((1,) + data.shape)

class DataBatch(object):
    """A data batch.

    MXNet's data iterator returns a batch of data for each `next` call.
    This data contains `batch_size` number of examples.

    If the input data consists of images, then shape of these images depend on
    the `layout` attribute of `DataDesc` object in `provide_data` parameter.

    If `layout` is set to 'NCHW' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, num_channel, height, width)``.
    If `layout` is set to 'NHWC' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, height, width, num_channel)``.
    The channels are often in RGB order.

    Parameters
    ----------
    data : list of `NDArray`, each array containing `batch_size` examples.
          A list of input data.
    label : list of `NDArray`, each array often containing a 1-dimensional array. optional
          A list of input labels.
    pad : int, optional
          The number of examples padded at the end of a batch. It is used when the
          total number of examples read is not divisible by the `batch_size`.
          These extra padded examples are ignored in prediction.
    index : numpy.array, optional
          The example indices in this batch.
    bucket_key : int, optional
          The bucket key, used for bucketing module.
    provide_data : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the data.
          The *i*-th element describes the name and shape of ``data[i]``.
    provide_label : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the label.
          The *i*-th element describes the name and shape of ``label[i]``.
    """
    def __init__(self, data, label=None, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        if data is not None:
            assert isinstance(data, (list, tuple)), "Data must be list of NDArrays"
        if label is not None:
            assert isinstance(label, (list, tuple)), "Label must be list of NDArrays"
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

    def __str__(self):
        data_shapes = [d.shape for d in self.data]
        if self.label:
            label_shapes = [l.shape for l in self.label]
        else:
            label_shapes = None
        return "{}: data shapes: {} label shapes: {}".format(
            self.__class__.__name__,
            data_shapes,
            label_shapes)

class DataIter(object):
    """The base class for an MXNet data iterator.

    All I/O in MXNet is handled by specializations of this class. Data iterators
    in MXNet are similar to standard-iterators in Python. On each call to `next`
    they return a `DataBatch` which represents the next batch of data. When
    there is no more data to return, it raises a `StopIteration` exception.

    Parameters
    ----------
    batch_size : int, optional
        The batch size, namely the number of items in the batch.

    See Also
    --------
    NDArrayIter : Data-iterator for MXNet NDArray or numpy-ndarray objects.
    CSVIter : Data-iterator for csv data.
    LibSVMIter : Data-iterator for libsvm data.
    ImageIter : Data-iterator for images.
    """
    def __init__(self, batch_size=0):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator to the begin of the data."""
        pass

    def next(self):
        """Get next data batch from iterator.

        Returns
        -------
        DataBatch
            The data of next batch.

        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Move to the next batch.

        Returns
        -------
        boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        list of NDArray
            The data of the current batch.
        """
        pass

    def getlabel(self):
        """Get label of the current batch.

        Returns
        -------
        list of NDArray
            The label of the current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The indices of examples in the current batch.
        """
        return None

    def getpad(self):
        """Get the number of padding examples in the current batch.

        Returns
        -------
        int
            Number of padding examples in the current batch.
        """
        pass

class CUB200Iter(DataIter):
    """Iterator for the CUB200-2011 dataset.
    Parameters
    ----------
    data_path : str,
        The path to dataset directory.
    batch_k : int,
        Number of images per class in a batch.
    batch_size : int,
        Batch size.
    batch_size : tupple,
        Data shape. E.g. (3, 224, 224).
    is_train : bool,
        Training data or testig data. Training batches are randomly sampled.
        Testing batches are loaded sequentially until reaching the end.
    """
    def __init__(self, data_path, batch_k, batch_size, data_shape, is_train):
        super(CUB200Iter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.batch_size = batch_size
        self.provide_data = [('data', self.data_shape)]
        self.batch_k = batch_k
        self.is_train = is_train

        self.train_image_files = [[] for _ in range(100)]
        self.test_image_files = []
        self.test_labels = []
        self.boxes = {}
        self.test_count = 0
        # for plotting and sampling --- only available for tests
        index = 0
        self.dict_classes = {}

        with open(os.path.join(data_path, 'images.txt'), 'r') as f_img, \
             open(os.path.join(data_path, 'image_class_labels.txt'), 'r') as f_label, \
             open(os.path.join(data_path, 'bounding_boxes.txt'), 'r') as f_box:
            for line_img, line_label, line_box in zip(f_img, f_label, f_box):
                fname = os.path.join(data_path, 'images', line_img.strip().split()[-1])
                label = int(line_label.strip().split()[-1]) - 1
                box = [int(float(v)) for v in line_box.split()[-4:]]
                self.boxes[fname] = box

                # Following "Deep Metric Learning via Lifted Structured Feature Embedding" paper,
                # we use the first 100 classes for training, and the remaining for testing.
                if label < 100:
                    self.train_image_files[label].append(fname)
                else:
                    self.test_labels.append(label)
                    self.test_image_files.append(fname)
                    self.dict_classes[index] = [label, fname]
                    index += 1

        self.n_test = len(self.test_image_files)

    def get_image(self, img, is_train):
        """Load and transform an image."""
        img_arr = cv2.imread(img)[...,::-1] 
        img_arr = transform(img_arr, 256, 256, is_train, self.boxes[img])
        return img_arr

    def sample_train_batch(self):
        """Sample a training batch (data and label)."""
        batch = []
        labels = []
        num_groups = self.batch_size // self.batch_k

        # For CUB200, we use the first 100 classes for training.
        sampled_classes = np.random.choice(100, num_groups, replace=False)
        for i in range(num_groups):
            img_fnames = np.random.choice(self.train_image_files[sampled_classes[i]],
                                          self.batch_k, replace=False)
            batch += [self.get_image(img_fname, is_train=True) for img_fname in img_fnames]
            labels += [sampled_classes[i] for _ in range(self.batch_k)]
        
        batch = np.concatenate(batch, axis=0).astype(np.float32)
        labels = np.array(labels)
        return Tensor(batch), Tensor(labels)

    def get_test_batch(self):
        """Sample a testing batch (data and label)."""

        batch_size = self.batch_size
        batch = [self.get_image(self.test_image_files[(self.test_count*batch_size + i)
                                                      % len(self.test_image_files)],
                                is_train=False) for i in range(batch_size)]
        labels = [self.test_labels[(self.test_count*batch_size + i)
                                   % len(self.test_image_files)] for i in range(batch_size)]
        
        batch = np.concatenate(batch, axis=0).astype(np.float32)
        labels = np.array(labels)
        return Tensor(batch), Tensor(labels)

    def reset(self):
        """Reset an iterator."""
        self.test_count = 0

    def next(self):
        """Return a batch."""
        if self.is_train:
            data, labels = self.sample_train_batch()
        else:
            if self.test_count * self.batch_size < len(self.test_image_files):
                data, labels = self.get_test_batch()
                self.test_count += 1
            else:
                self.test_count = 0
                raise StopIteration
        return DataBatch(data=[data], label=[labels])

def cub200_iterator(data_path, batch_k, batch_size, data_shape):
    """Return training and testing iterator for the CUB200-2011 dataset."""
    return (CUB200Iter(data_path, batch_k, batch_size, data_shape, is_train=True),
            CUB200Iter(data_path, batch_k, batch_size, data_shape, is_train=False))