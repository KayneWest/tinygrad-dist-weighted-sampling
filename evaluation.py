# pylint: disable=E0401
"""
methods for evaluating the effectiveness of the model
"""
import random
import time

import faiss
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
matplotlib.use('Agg')

def get_distance_np(x):
  sim = np.dot(x, x.T)
  dist = 2 - 2 * sim
  dist = dist + np.eye(dist.shape[0])
  dist = np.sqrt(dist)
  return dist

def evaluate_emb(emb, labels):
  """Evaluate embeddings based on Recall@k."""
  d_mat = get_distance_np(emb)
  names = []
  accs = []
  for k in [1, 2, 4, 8, 16]:
    names.append('Recall@%d' % k)
    correct, cnt = 0.0, 0.0
    for i in range(emb.shape[0]):
      d_mat[i, i] = 1e10
      nns = argpartition(d_mat[i], k)[:k]
      if any(labels[i] == labels[nn] for nn in nns):
        correct += 1
      cnt += 1
    accs.append(correct/cnt)
  return names, accs

#
def test(net, val_data):
  """Test a model."""
  val_data.reset()
  outputs = []
  labels = []
  for batch in val_data:
    data = batch.data[0]
    label = batch.label[0]

    embeddings = net.encode(data)
    outputs.append(embeddings.numpy())
    labels.append(label.numpy())

  outputs = np.concatenate(outputs, axis=0)[:val_data.n_test]
  labels = np.concatenate(labels, axis=0)[:val_data.n_test]
  return evaluate_emb(outputs, labels)

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

# pylint: disable=R0914,R0913,C0103
def plot_similarity_query(index, val_data, embedding, img, fname,
              epoch, save_dir, save_model_prefix):
  """
  method for plotting the similarity queries of the validation
  data

  Args:
  -----
  index: faiss.Index
    the entire index you're using to evaluate your dataset
  val_data: mx.io.DataIter
    the data iter from the data.data directory
  embedding: np.array
    the query image embedded
  img: np.array
    the raw, jpg image you're querying for
  fname: str
    the filepath of img
  epoch: int
    the current training epoch
  save_dir: str
    the directory you're saving this plot to
  save_model_prefix: str
    the prefix of the model you're training
    this is used to prefix the image being saved
  """

  real_label = fname.split('/')[-2]
  knn = index.search(embedding, 6)

  m_0 = cv2.imread(val_data.dict_classes[knn[1][0][1]][1])[..., ::-1]
  m_1 = cv2.imread(val_data.dict_classes[knn[1][0][2]][1])[..., ::-1]
  m_2 = cv2.imread(val_data.dict_classes[knn[1][0][3]][1])[..., ::-1]
  m_3 = cv2.imread(val_data.dict_classes[knn[1][0][4]][1])[..., ::-1]
  query_img = resize_short_within(img[..., ::-1], short=416, max_size=1024, mult_base=32)
  m_0 = resize_short_within(m_0, short=416, max_size=1024, mult_base=32)
  m_1 = resize_short_within(m_1, short=416, max_size=1024, mult_base=32)
  m_2 = resize_short_within(m_2, short=416, max_size=1024, mult_base=32)
  m_3 = resize_short_within(m_3, short=416, max_size=1024, mult_base=32)

  _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4),
               sharex=True, sharey=True)
  ax = axes.ravel()

  label = 'L2 Distance: {:.2f}'

  ax[0].imshow(query_img)
  ax[0].set_xlabel(label.format(0.0))
  ax[0].set_title('Original Query Image')

  ax[1].imshow(m_0)
  ax[1].set_xlabel(label.format(knn[0][0][0]))
  ax[1].set_title('Nearest Result - 1')

  ax[2].imshow(m_1)
  ax[2].set_xlabel(label.format(knn[0][0][1]))
  ax[2].set_title('Nearest Result - 2')

  ax[3].imshow(m_2)
  ax[3].set_xlabel(label.format(knn[0][0][2]))
  ax[3].set_title('Nearest Result - 3')

  ax[4].imshow(m_3)
  ax[4].set_xlabel(label.format(knn[0][0][3]))
  ax[4].set_title('Nearest Result - 4')

  plt.tight_layout()
  plt.savefig("{}/{}-EPOCH-{}-{}.png".format(save_dir, save_model_prefix, epoch, real_label))

# pylint: disable=R0912, R0915, C0325
def evaluate_emb_faiss(
  emb, labels, val_data, save_dir,
  save_model_prefix, epoch=0, 
  plot=True):
  """Evaluate embeddings based on Recall@k.

  Args:
  -----
  emb: np.ndarray
    the entire validation dataset as a numpy array
  labels: np.ndarry
    the labels of the validation dataset
  val_data: mx.io.DataIter
    the data iter that contains our validation data
    used to sample random images from
  save_dir: str
    the directory where we save our images
  save_model_prefix: str
    the prefix of the model you're training
    this is used to prefix the image being saved
  epoch: int
    the epoch for plotting

  Returns:
  -------
  names: tuple(str)
    the name of the recall metrics (Recall@k)
  accs: tuple(floats)
    the recall information regarding recall@k
  """
  print('creating index')
  index = faiss.IndexFlatL2(emb.shape[1])
  index.add(emb)
  print('plotting imgs')
  if plot:
    for _ in range(10):
      idx = random.choice(list(range(emb.shape[0])))
      fname = val_data.dict_classes[idx][1]
      img = cv2.imread(fname)
      embedding = emb[idx].reshape(1, -1)
      plot_similarity_query(index, val_data, embedding,
                  img, fname, epoch, save_dir, save_model_prefix)

  names = []
  accs = []
  ps = faiss.ParameterSpace()
  ps.initialize(index)
  t0 = time.time()
  _, I = index.search(emb, 100)
  for rank in 1, 2, 4, 8, 16:
    names.append('Recall@%d' % rank)
    correct, cnt = 0.0, 0.0
    for i in range(I.shape[0]):
      nns = I[i][1:rank+1]
      if any(labels[i] == labels[nn] for nn in nns):
        correct += 1
      cnt += 1
    accs.append(correct/cnt)
  t1 = time.time()
  print ("Time: %8.3f" % (((t1 - t0) * 1000.0) / emb.shape[1]))
  return names, accs

def test_faiss(net, val_data, epoch, save_dir='', save_model_prefix=''):
  """Test a model."""
  print('starting test')
  val_data.reset()
  val_data.reset()
  outputs = []
  labels = []
  count = 0
  for batch in val_data:
    data = batch.data[0]
    label = batch.label[0]
    outputs.append(net(data).numpy())
    
    labels.append(label.numpy())
    count += 1
  print('finished iterating through val_data')
  outputs = np.vstack(outputs)
  labels = np.hstack(labels)
  labels = labels.reshape(labels.shape[0], 1)
  if labels.shape[0] != outputs.shape[0]:
    if labels.shape[0] > outputs.shape[0]:
      labels = labels[:outputs.shape[0]]
    else:
      outputs = outputs[:labels.shape[0]]
  val_data.reset()
  return evaluate_emb_faiss(outputs, labels, val_data,
                save_dir, save_model_prefix,
                epoch=epoch)
