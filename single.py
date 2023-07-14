import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tinygrad.state import get_parameters
from resnet import ResNet50

class MarginNet:
  def __init__(self, emb_dim, batch_size, batch_k, cutoff=0.5, non_zero_cuttoff=1.4, sample=True, **kwargs):
    super(MarginNet, self).__init__()
    self.feature_net = ResNet50()
    self.dense = Tensor.kaiming_uniform(2048, emb_dim)
    self.batch_size = batch_size
    self.sampler = DistanceWeightedSampling(batch_size, batch_k, cutoff, non_zero_cuttoff, **kwargs) if sample else None

  def feature_detector(self, x):
    out = self.feature_net.bn1(self.feature_net.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.feature_net.layer1)
    out = out.sequential(self.feature_net.layer2)
    out = out.sequential(self.feature_net.layer3)
    out = out.sequential(self.feature_net.layer4)
    out = out.avg_pool2d(kernel_size=(7,7)) # equiv of AdaptiveAvgPool
    return out.reshape(self.batch_size, -1) # -1?

  def load_basenet_features(self): 
    self.feature_net.load_from_pretrained()
    del self.feature_net.fc # del last layer

  def __call__(self, x):
    z = self.feature_detector(x)
    z = Tensor.linear(z, self.dense)
    if z.shape: z = z.reshape((z.shape[0],-1))
    normalized_emb = z / (z ** 2).sum() + 1e-5).sqrt()
    return normalized_emb

  def sample(self, x):
    z = self.feature_detector(x)
    z = Tensor.linear(z, self.dense)
    if z.shape: z = z.reshape((z.shape[0],-1))
    normalized_emb = z / (z ** 2).sum() + 1e-5).sqrt()
    a_indices, p_indices, n_indices = self.sampler(normalized_emb)
    return a_indices, p_indices, n_indices, normalized_emb

class DistanceWeightedSampling:
  def __init__(self, batch_size, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
    super(DistanceWeightedSampling, self).__init__()
    self.batch_k = batch_k
    self.cutoff = cutoff
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
  
    # fill mask matrix and 
    mask = np.ones((batch_size, batch_size)).astype(np.float32)
    for i in range(0, batch_size, batch_k): mask[i:i + batch_k, i:i + batch_k] = 0.
    self.mask_uniform_probs = mask * (1.0 / (batch_size - batch_k))
    self.mask = Tensor(mask.astype(np.float32), requires_grad=False)

  def get_weights(self, x):
    # get distance matrix
    square = (x**2).sum(axis=1, keepdim=True)
    distance = (square + square.transpose() - (2.0 * x.dot(x.transpose())) + Tensor.eye(x.shape[0])).sqrt().maximum(self.cutoff)
    # Subtract max(log(distance)) for stability.
    log_weights = ((2.0 - float(x.shape[1])) * distance.log() - (float(x.shape[1] - 3) / 2) * (1.0 - 0.25 * (distance ** 2.0)).log())
    weights = (log_weights - log_weights.max()).exp()
    # get weights
    weights = weights * self.mask * (distance < self.nonzero_loss_cutoff)
    weights_sum = Tensor.sum(weights, axis=1, keepdim=True)
    weights = weights / weights_sum
    return weights, weights_sum

  # needs numpy
  def get_samples(self, weights, weights_sum):
    a_indices = []
    p_indices = []
    n_indices = []
    np_weights = weights.realize().numpy()
    np_weights_sum = weights_sum.realize().numpy()

    for i in range(np_weights.shape[0]):
      block_idx = i // self.batch_k
      
      if np_weights_sum[i] != 0:
        try:
          n_indices += np.random.choice(np_weights.shape[0], self.batch_k-1, p=np_weights[i]).tolist()
        except ValueError: #ValueError: probabilities do not sum to 1 
          idx = np.argmin(np_weights[i])
          np_weights[i][idx] += 1 - np_weights[i].sum()
          n_indices +=  np.random.choice(np_weights.shape[0], self.batch_k-1, p=np_weights[i]).tolist()
      else:
        n_indices +=  np.random.choice(np_weights.shape[0], self.batch_k-1, p=self.mask_uniform_probs[i]).tolist()
      for j in range(block_idx * self.batch_k, (block_idx + 1)*self.batch_k):
        if j != i:
          a_indices.append(i)
          p_indices.append(j)
    return a_indices, p_indices, n_indices

  def __call__(self, x):
    weights, weights_sum = self.get_weights(x)
    a_indices, p_indices, n_indices = self.get_samples(weights, weights_sum)
    return a_indices, p_indices, n_indices


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
  def __init__(self, batch_size, batch_k, embed_dim, margin=0.2, nu=0.0, **kwargs):
    super(MarginLoss, self).__init__()
    self._margin = margin
    self._nu = nu
    self.batch_size = batch_size
    self.batch_k = batch_k
    self.embed_dim = embed_dim

  def get_losses(self, x, y, beta, a_indices, p_indices, n_indices):
    y_r = y.realize().numpy()
    anchors = Tensor.cat(*[x[a] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    positives = Tensor.cat(*[x[p] for p in p_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    negatives = Tensor.cat(*[x[n] for n in n_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    betas = Tensor.cat(*[beta[y_r[a]] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), 1)

    beta_reg_loss = betas.sum() * self._nu

    d_ap = ((positives - anchors).square().sum(axis=1) + 1e-8).sqrt()
    d_an = ((negatives - anchors).square().sum(axis=1) + 1e-8).sqrt()

    # max is 0.0
    pos_loss = (d_ap - betas + self._margin).maximum(0.)
    neg_loss = (betas - d_an + self._margin).maximum(0.)

    # RuntimeError: backward not implemented for <class 'tinygrad.mlops.Equal'>
    pair_cnt = float(Tensor.sum((pos_loss > 0.0) + (neg_loss > 0.0)).numpy())

    loss = ((pos_loss + neg_loss).sum() + beta_reg_loss) / pair_cnt
    return loss

  def losses(self, x, y, beta_in, a_indices, p_indices, n_indices):
    total_loss = Tensor(0.0)
    pair_cnt = Tensor(0)
    beta_reg_loss = Tensor(0)
    y_r = y.realize().numpy()
    for i in range(len(a_indices)):
      i_beta = beta_in[y_r[a_indices[i]]] # this is inc
      i_anchors = x[a_indices[i]]
      i_positives = x[p_indices[i]]
      i_negatives = x[n_indices[i]] 
      d_ap = Tensor.sqrt(Tensor.sum(Tensor.square(i_positives - i_anchors)) + 1e-8)
      d_an = Tensor.sqrt(Tensor.sum(Tensor.square(i_negatives - i_anchors)) + 1e-8)

      # max is 0.0 
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
    return loss

  # loss function w y already realized
  def get_losses_(self, x, y_r, beta, a_indices, p_indices, n_indices):
    anchors = Tensor.cat(*[x[a] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    positives = Tensor.cat(*[x[p] for p in p_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    negatives = Tensor.cat(*[x[n] for n in n_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    betas = Tensor.cat(*[beta[y_r[a]] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), 1)

    beta_reg_loss = betas.sum() * self._nu

    d_ap = ((positives - anchors).square().sum(axis=1) + 1e-8).sqrt()
    d_an = ((negatives - anchors).square().sum(axis=1) + 1e-8).sqrt()

    # max is 0.0
    pos_loss = (d_ap - betas + self._margin).maximum(0.)
    neg_loss = (betas - d_an + self._margin).maximum(0.)

    # RuntimeError: backward not implemented for <class 'tinygrad.mlops.Equal'>
    pair_cnt = float(Tensor.sum((pos_loss > 0.0) + (neg_loss > 0.0)).numpy())

    loss = ((pos_loss + neg_loss).sum() + beta_reg_loss) / pair_cnt
    return loss

  def __call__(self, x, y, beta_in, a_indices, p_indices, n_indices):
    return self.get_losses_(x, y, beta_in, a_indices, p_indices, n_indices)
