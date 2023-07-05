# pylint: disable=E0401
"""
helper functioons for manual debugging
"""
import random
import cv2
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from mxnet import nd

NORMALIZE_FN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

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
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
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
    # pylint: disable=C0103
    if interp == 9:
        # pylint: disable=R1705
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            # pylint: disable=R1705
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
        raise ValueError('Unknown interp method %d' % interp)
    return interp

# pylint: disable=C0111, C0103, C0301, R0913
def resize_short_within(img, short=512, max_size=1024, mult_base=32, interp=2, debug=False):
    """
    similar function to gluoncv's resize short within code that resizes
    the short side of the image and scales the long side to that new short size

    Args:
    -----
    img: np.array
        the image you want to resize
    short: int
        the desired short size
    max_size: int
        the maximum size the short side can take
    mult_base: int
        how to scale the resizing
    interp: int
        the interpretation method used by _get_interp_method
    debug: Bool
        prints the output of the new-size
    Returns:
    --------
    img: np.array
        the resized image
    """
    h, w, _ = img.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base), int(np.round(h * scale / mult_base) * mult_base))
    if debug:
        print(new_w, new_h)
    img = cv2.resize(img, (new_w, new_h), interpolation=_get_interp_method(interp, (h, w, new_h, new_w)))
    return img

def resize_and_pad(im, desired_size, pad):
    """
    function to resize a given image and pad it

    Args:
    -----
    im : numpy.array
        the image that you want to resize and pad
    desired_size: int
        the h/w size (photo will be sqaure)
    pad: bool
        whether to pad the image or not
    Returns:
    --------
    new_im: the resized and padded image
    """
    if desired_size:
        desired_size = int(desired_size)
    else:
        desired_size = max(im.shape)

    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    if not pad:
        # new_size should be in (width, height) format
        im = cv2.resize(im, (desired_size, desired_size))
        return im
    else:

        im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im

# pylint: disable=R0912
def transform(data, target_size, is_train, box, mult, rotator, crop=False):
    """
    function to crop and normalize an array
    Crop and normnalize an image nd array.

    Args:
    -----
    data: np.array
        image you want to be transformed
    target_size: int
        the size you want to transform to
    is_train: bool
        whether or not to transform the image
    box: list
        the xmin, ymin, xmax, ymax of the bounding box
        location in the image
    mult: int
        the multiplier to aaugment the image size
    rotator: ImageAugmentor
        the class that does the image-augmentor
    crop: bool
        whether or not to crop the image
    Returns:
    --------
    data: mx.nd.array
        the transformed image
    """

    crop_shape = int(target_size / mult)
    if box is not None:
        #x, y, w, h = box
        xmin, ymin, xmax, ymax = box
        new_data = data.asnumpy()[ymin:ymax, xmin:xmax]

        # print(data.shape)
        if new_data.shape[0] == 0 or new_data.shape[1] == 0:
            ymin = min(ymin, data.shape[0]-1)
            ymax = min(ymax, data.shape[0]-1)
            xmin = min(xmin, data.shape[1]-1)
            xmax = min(xmax, data.shape[1]-1)
            data = data[ymin:ymax, xmin:xmax]
        else:
            data = new_data

    # retain aspect ration
    if isinstance(data, mx.nd.NDArray):
        data = data.asnumpy()

    data = resize_and_pad(data, target_size, pad=True)

    if is_train:
        if rotator is not None:
            if random.random() < 0.5:
                data = rotator.augment_image(data)
    data = mx.nd.array(data)

    data = NORMALIZE_FN(data)

    # Resize to target_size * target_ht.
    # data = mx.image.imresize(data, target_size, target_ht)

    # Normalize in the same way as the pre-trained model.
    ## data = data.astype(np.float32) / 255.0
    ## data = (data - mx.nd.array([0.485, 0.456, 0.406])) / mx.nd.array([0.229, 0.224, 0.225])

    # I think flip for beers isn't necessary
    if not crop and rotator is None:
        crop = True

    if crop:
        if is_train:
            if random.random() < 0.5:
                data = nd.flip(data, axis=1)
            data, _ = mx.image.random_crop(data, (crop_shape, crop_shape))
        else:
            data, _ = mx.image.center_crop(data, (crop_shape, crop_shape))

    # Transpose from (target_size, target_ht, 3)
    # to (3, target_size, target_ht).
    # data = nd.transpose(data, (2, 0, 1))

    # If image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))

    data = data.expand_dims(0)
    return data #data.reshape((1,) + data.shape)