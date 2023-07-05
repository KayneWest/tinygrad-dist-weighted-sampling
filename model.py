

import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tinygrad.state import get_parameters

GPU = getenv("GPU")
QUICK = getenv("QUICK")
DEBUG = getenv("DEBUG")

def l2_norm(x):
  if len(x.shape):
    x = x.reshape((x.shape[0],-1))
  out = x / Tensor.sqrt(Tensor.sum(x ** 2) + 1e-5)
  return out

def get_distance(x):
  sim = Tensor.dot(x, x.T)
  dist = 2 - 2 * sim
  dist = dist + Tensor.eye(dist.shape[0])
  dist = dist.sqrt()
  return dist

def clamp(x, max_value=None, min_value=None):
  if not min_value:
    min_value = x.min()
  else:
    if not isinstance(min_value, Tensor):
      min_value = Tensor(min_value)
  if not max_value:
    max_value = x.max()
    if not isinstance(max_value, Tensor):
      max_value = Tensor(max_value)
  return Tensor.min(Tensor.max(x, min_value), max_value) 

class ResNetFeats:
  def __init__(self, net, batch_size=1, kernel_size=(7,7)) -> None:
    self.net = net
    # del the fc layer
    del self.net.fc
    self.batch_size = batch_size
  def __call__(self, x):
    out = self.net.bn1(self.net.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.net.layer1)
    out = out.sequential(self.net.layer2)
    out = out.sequential(self.net.layer3)
    out = out.sequential(self.net.layer4)
    # add equiv of AdaptiveAvgPool
    out = out.avg_pool2d(kernel_size=(7,7))
    return out.reshape(self.batch_size,-1)

class MarginNet:
  r"""Embedding network with distance weighted sampling.
  It takes a base CNN and adds an embedding layer and a
  sampling layer at the end.

  Parameters
  ----------
  base_net : Block
    Base network.
  emb_dim : int
    Dimensionality of the embedding.
  batch_k : int
    Number of images per class in a batch. Used in sampling.

  Inputs:
    - **data**: input tensor with shape (batch_size, channels, width, height).
    Here we assume the consecutive batch_k images are of the same class.
    For example, if batch_k = 5, the first 5 images belong to the same class,
    6th-10th images belong to another class, etc.

  Outputs:
    - The normalized embedding.
  """
  
  def __init__(self, base_net, base_net_out, emb_dim, batch_k, feat_dim=None, normalize=False):
    super(MarginNet, self).__init__()
    self.base_net = base_net
    self.dense = Tensor.kaiming_uniform(base_net_out, emb_dim)
    self.normalize = l2_norm

  def encode(self, x):
    """
    model used in forward function without training or sampling,
    this method is used in the beer-service

    Args:
    -----
    x : mx.nd.array
      the data you want to embed
    Returns:
    --------
    z: mx.nd.array
      the embedded data
    """
    z = self.base_net(x)
    z = Tensor.linear(z, self.dense)
    z = self.normalize(z)
    return z

  def save(self, filename):
    with open(filename+'.npy', 'wb') as f:
      for par in get_parameters(self):
        np.save(f, par.cpu().numpy())

  def load(self, filename):
    with open(filename+'.npy', 'rb') as f:
      for par in get_parameters(self):
        try:
          par.cpu().numpy()[:] = np.load(f)
          if GPU:
            par.gpu()
        except:
          print('Could not load parameter')

  def __call__(self, x):
    return self.encode(x)

# pylint: disable=R0903
class DistanceWeightedMarginLoss:
  r"""Margin based loss.

  Parameters
  ----------
  margin : float
    Margin between positive and negative pairs.
  nu : float
    Regularization parameter for beta.

  Inputs:
    - anchors: sampled anchor embeddings.
    - positives: sampled positive embeddings.
    - negatives: sampled negative embeddings.
    - beta_in: class-specific betas.
    - a_indices: indices of anchors. Used to get class-specific beta.

  Outputs:
    - Loss.
  """
  def __init__(self, batch_size, batch_k, margin=0.2, nu=0.0, weight=None, batch_axis=0, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False, **kwargs):
    super(DistanceWeightedMarginLoss, self).__init__()
    self._margin = margin
    self._nu = nu
    self.batch_k = batch_k
    self.cutoff = cutoff
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
    self.normalize = normalize

    mask = np.ones((batch_size, batch_size))
    for i in range(0,batch_size, batch_k):
      mask[i:i+batch_k, i:i+batch_k] = float(0)

    self.mask_uniform_probs = mask *(1.0/(batch_size-batch_k))
    self.mask = Tensor(mask)

  # pylint: disable=C0111,R0914,W0221,R0913
  def forward(self, x, y, beta_in):
    """
    deriving the loss for the model's outputs

    Args:
    -----
    x: Tensor
    y: Tensor
    beta_in: Tensor

    Returns:
    --------
    loss: Tensor
    pair_cnt: Tensor
    """
    k = self.batch_k
    n, d = x.shape
    distance = get_distance(x)
    distance = Tensor.maximum(distance, self.cutoff)

    # Subtract max(log(distance)) for stability.
    log_weights = ((2.0 - float(d)) * Tensor.log(distance)
             - (float(d - 3) / 2) * Tensor.log(1.0 - 0.25 * (distance ** 2.0)))
    weights = Tensor.exp(log_weights - Tensor.max(log_weights))
    # grad works here

    if x.device != weights.device:
      weights = weights.to(x.device)

    weights = weights*self.mask*((distance < self.nonzero_loss_cutoff)) + 1e-8

    weights_sum = Tensor.sum(weights, axis=1, keepdim=True)
    weights = weights / weights_sum

    a_indices = []
    p_indices = []
    n_indices = []

    np_weights = weights.detach().numpy()
    for i in range(n):
      block_idx = i // k

      if weights_sum[i] != 0:
        try:
          n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
        except ValueError: #ValueError: probabilities do not sum to 1 
          to_add = 1-np_weights[i].sum()
          idx = np.argmin(np_weights[i])
          np_weights[i][idx]+=to_add
          n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
      else:
        n_indices +=  np.random.choice(n, k-1, p=self.mask_uniform_probs[i]).tolist()
      for j in range(block_idx * k, (block_idx + 1)*k):
        if j != i:
          a_indices.append(i)
          p_indices.append(j)

    total_loss = Tensor(0.0)
    pair_cnt = Tensor(0)
    beta_reg_loss = Tensor(0)
    for i in range(len(a_indices)):
      i_beta = beta_in[i]
      i_anchors = x[a_indices[i]]
      i_positives = x[p_indices[i]]
      i_negatives = x[n_indices[i]]
      d_ap = Tensor.sqrt(Tensor.sum(Tensor.square(i_positives - i_anchors), axis=0) + 1e-8)
      d_an = Tensor.sqrt(Tensor.sum(Tensor.square(i_negatives - i_anchors), axis=0) + 1e-8)

      pos_loss = Tensor.maximum(d_ap - i_beta + self._margin, Tensor(0.0))
      neg_loss = Tensor.maximum(i_beta - d_an + self._margin, Tensor(0.0))

      total_loss = total_loss + (pos_loss + neg_loss)

      if pos_loss > 0.0:
        pair_cnt = pair_cnt + Tensor(1)

      if neg_loss > 0.0:
        pair_cnt = pair_cnt + Tensor(1)

      beta_reg_loss = beta_reg_loss + i_beta

    beta_reg_loss = beta_reg_loss * self._nu
    if pair_cnt == 0.0:
      loss = Tensor.sum(total_loss)
    else:
      loss = (Tensor.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt
    return loss, pair_cnt

  def __call__(self, x, y, beta_in):
    return self.forward(x, y, beta_in)

# pylint: disable=R0903
class MarginLoss:
  r"""Margin based loss.

  Parameters
  ----------
  margin : float
    Margin between positive and negative pairs.
  nu : float
    Regularization parameter for beta.

  Inputs:
    - anchors: sampled anchor embeddings.
    - positives: sampled positive embeddings.
    - negatives: sampled negative embeddings.
    - beta_in: class-specific betas.
    - a_indices: indices of anchors. Used to get class-specific beta.

  Outputs:
    - Loss.
  """
  def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
    super(MarginLoss, self).__init__()
    self._margin = margin
    self._nu = nu

  # pylint: disable=C0111,R0914,W0221,R0913
  def foward(self, anchors, positives, negatives, beta_in, a_indices=None):
    """
    deriving the loss for the model's outputs

    Args:
    -----
    anchors: mx.nd.array
    positives: mx.nd.array
    negatives: mx.nd.array
    beta_in: float
    a_indices: bool

    Returns:
    --------
    """
    if a_indices is not None:
      # Jointly train class-specific beta.
      beta = Tensor(beta_in.numpy()[a_indices])
      beta_reg_loss = Tensor.sum(beta) * self._nu
    else:
      # Use a constant beta.
      beta = beta_in
      beta_reg_loss = 0.0
    
    if isinstance(beta, np.ndarray):
      beta = Tensor(beta)

    d_ap = Tensor.sqrt(Tensor.sum(Tensor.square(positives - anchors), axis=1) + 1e-8)
    d_an = Tensor.sqrt(Tensor.sum(Tensor.square(negatives - anchors), axis=1) + 1e-8)

    pos_loss = Tensor.maximum(d_ap - beta + self._margin, Tensor(0.0))
    neg_loss = Tensor.maximum(beta - d_an + self._margin, Tensor(0.0))

    pair_cnt = Tensor.sum((pos_loss > 0.0) + (neg_loss > 0.0))
    if pair_cnt == 0.0:
      # When poss_loss and neg_loss is zero then total loss is zero as well
      loss = Tensor.sum(pos_loss + neg_loss)
    else:
      # Normalize based on the number of pairs.
      loss = (Tensor.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt
    # pylint: disable=W0212
    return loss, pair_cnt

class DistanceWeightedSampling:
  '''
  parameters
  ----------
  batch_k: int
    number of images per class

  Inputs:
    data: input tensor with shapeee (batch_size, edbed_dim)
      Here we assume the consecutive batch_k examples are of the same class.
      For example, if batch_k = 5, the first 5 examples belong to the same class,
      6th-10th examples belong to another class, etc.
  Outputs:
    a_indices: indicess of anchors
    x[a_indices]
    x[p_indices]
    x[n_indices]
    xxx
  '''

  def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False,  **kwargs):
    super(DistanceWeightedSampling,self).__init__()
    self.batch_k = batch_k
    self.cutoff = cutoff
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
    self.normalize = normalize

  def forward(self, x):
    k = self.batch_k
    n, d = x.shape
    eye_tensor = Tensor.eye(n)
    # distance = get_distance(x, is_train=True, eye_tensor=eye_tensor)
    distance = get_distance(x)
    distance = Tensor.maximum(distance, self.cutoff)

    # Subtract max(log(distance)) for stability.
    log_weights = ((2.0 - float(d)) * Tensor.log(distance)
             - (float(d - 3) / 2) * Tensor.log(1.0 - 0.25 * (distance ** 2.0)))
    weights = Tensor.exp(log_weights - Tensor.max(log_weights))

    if x.device != weights.device:
      weights = weights.to(x.device)

    mask = np.ones(weights.shape)
    for i in range(0,n,k):
      mask[i:i+k, i:i+k] = 0

    mask_uniform_probs = mask *(1.0/(n-k))
    mask = Tensor(mask)

    weights = weights*mask*((distance < self.nonzero_loss_cutoff)) + 1e-8
    weights_sum = Tensor.sum(weights, axis=1, keepdim=True)
    weights = weights / weights_sum

    a_indices = []
    p_indices = []
    n_indices = []

    np_weights = weights.cpu().numpy()
    for i in range(n):
      block_idx = i // k

      if weights_sum[i] != 0:
        n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
      else:
        n_indices +=  np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
      for j in range(block_idx * k, (block_idx + 1)*k):
        if j != i:
          a_indices.append(i)
          p_indices.append(j)
    
    realized = x.numpy()
    return  a_indices, Tensor(realized[a_indices]), Tensor(realized[p_indices]), Tensor(realized[n_indices]), x

