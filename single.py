import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tinygrad.state import get_parameters
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from models.resnet import ResNet50
from extra.datasets import fetch_cifar

# def _tri(r:int, c:int, k:int=0, **kwargs) -> Tensor: return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r,c) <= Tensor.arange(c-k, start=-k, **kwargs).unsqueeze(0).expand(r,c)
# def triu(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k, dtype=self.dtype).where(self, Tensor.zeros_like(self))
# def tril(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k+1, dtype=self.dtype).where(Tensor.zeros_like(self), self)

def fetch_cifar(train=True):
  cifar10_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618], dtype=np.float32).reshape(1,3,1,1)
  cifar10_std = np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628], dtype=np.float32).reshape(1,3,1,1)
  fn = os.path.dirname(__file__)+"/cifar-10-python.tar.gz"
  download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', fn)
  tt = tarfile.open(fn, mode='r:gz')
  if train:
    db = [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)]
  else:
    db = [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")]
  X = np.concatenate([x[b'data'].reshape((-1, 3, 32, 32)) for x in db], axis=0)
  Y = np.concatenate([np.array(x[b'labels']) for x in db], axis=0)
  X = X / 255.
  X = (X - cifar10_mean) / cifar10_std
  return X, Y

class OptimizerGroup(list):
  def zero_grad(self): [optimizer.zero_grad() for optimizer in self]
  def step(self): [optimizer.step() for optimizer in self]

class MarginNet:
  def __init__(self, emb_dim, batch_size, batch_k, cutoff=0.5, non_zero_cuttoff=1.4):
    super(MarginNet, self).__init__()
    self.batch_size = batch_size
    self.feature_net = ResNet50()
    self.dense = nn.Linear(2048, emb_dim)
    self.sampler = DistanceWeightedSampling(batch_size, batch_k, emb_dim, cutoff, non_zero_cuttoff)

  def feature_detector(self, x):
    out = self.feature_net.bn1(self.feature_net.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.feature_net.layer1)
    out = out.sequential(self.feature_net.layer2)
    out = out.sequential(self.feature_net.layer3)
    out = out.sequential(self.feature_net.layer4)
    out = out.avg_pool2d(kernel_size=(7,7)) # equiv of AdaptiveAvgPool
    return out.reshape(self.batch_size, -1)

  def load_basenet_features(self): 
    self.feature_net.load_from_pretrained()
    del self.feature_net.fc # del last layer

  def inference(self, x): 
    z = self.feature_detector(x)
    z = self.dense(z)
    if z.shape: z = z.reshape((z.shape[0],-1))
    normalized_emb = z / Tensor.sqrt(Tensor.sum(z ** 2) + 1e-5)
    return normalized_emb

  def sample(self, x):
    return self.sampler(self.inference(x))

class DistanceWeightedSampling:
  def __init__(self, batch_size, batch_k, embed_dim, cutoff=0.5, nonzero_loss_cutoff=1.4):
    super(DistanceWeightedSampling, self).__init__()
    self.batch_k = batch_k
    self.cutoff = cutoff
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
    self.embed_dim = embed_dim
    self.batch_size = batch_size
    mask = np.ones((batch_size, batch_size)).astype(np.float32)
    for i in range(0, batch_size, self.batch_k): mask[i:i+self.batch_k, i:i+self.batch_k] = 0
    self.mask = Tensor(mask, requires_grad=False)

  def __call__(self, x):
    square = (x**2).sum(axis=1, keepdim=True)
    distance = (square + square.transpose() - (2.0 * x.dot(x.transpose())) + Tensor.eye(x.shape[0])).sqrt().maximum(self.cutoff)
    log_weights = ((2.0 - float(x.shape[1])) * distance.log() - (float(x.shape[1] - 3) / 2) * (1.0 - 0.25 * (distance ** 2.0)).log())
    weights = (log_weights - log_weights.max()).exp()
    weights = weights * self.mask * (distance < self.nonzero_loss_cutoff)
    weights_sum = Tensor.sum(weights, axis=1, keepdim=True)
    weights = weights / weights_sum
    np_weights = weights.realize().numpy()
    # sample space
    a_indices = []
    p_indices = []
    n_indices = []
    for i in range(np_weights.shape[0]):
        block_idx = i // self.batch_k
        try:
            n_indices += np.random.choice(np_weights.shape[0], self.batch_k-1, p=np_weights[i]).tolist()
        except:
            n_indices += np.random.choice(np_weights.shape[0], self.batch_k-1).tolist()
        for j in range(block_idx * self.batch_k, (block_idx + 1) * self.batch_k):
            if j != i:
                a_indices.append(i)
                p_indices.append(j)
    anchors = Tensor.cat(*[x[a] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    positives = Tensor.cat(*[x[p] for p in p_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    negatives = Tensor.cat(*[x[n] for n in n_indices]).reshape(self.batch_size * (self.batch_k - 1), self.embed_dim)
    return a_indices, anchors, positives, negatives, x

class MarginLoss:
  def __init__(self, batch_size, batch_k, embed_dim, margin=0.2, nu=0.0, **kwargs):
    super(MarginLoss, self).__init__()
    self._margin = margin
    self._nu = nu
    self.batch_size = batch_size
    self.batch_k = batch_k
    self.embed_dim = embed_dim

  def __call__(self, y_r, beta, a_indices, anchors, positives, negatives):
    betas = Tensor.cat(*[beta[y_r[a]] for a in a_indices]).reshape(self.batch_size * (self.batch_k - 1), 1)
    beta_reg_loss = betas.sum() * self._nu    
    d_ap = ((positives - anchors).square().sum(axis=1) + 1e-8).sqrt()
    d_an = ((negatives - anchors).square().sum(axis=1) + 1e-8).sqrt()
    pos_loss = (d_ap - betas + self._margin).maximum(0.)
    neg_loss = (betas - d_an + self._margin).maximum(0.)
    pair_cnt = float(Tensor.sum((pos_loss > 0.0) + (neg_loss > 0.0)).numpy())
    loss = ((pos_loss + neg_loss).sum() + beta_reg_loss) / pair_cnt
    return loss

class CiFarSampler:
  def __init__(self, batch_size, batch_k, classes=5):
    self.batch_size = batch_size
    self.batch_k = batch_k
    self.num_groups = self.batch_size // self.batch_k
    self.classes = classes
    # unfuck cifar...
    
    X_train, Y_train = fetch_cifar(train=True)
    X_test, Y_test = fetch_cifar(train=False)
    features = np.vstack([X_train, X_test])
    labels = np.hstack([Y_train, Y_test])
    self.X_train = features[labels < classes]
    self.Y_train = labels[labels < classes]
    self.X_test = features[labels >= classes]
    self.Y_test = labels[labels >= classes]
    del features
    del labels

  def gen_sample_train(self):
    # For CUB200, we use the first 100 classes for training.
    sampled_classes = np.random.choice(self.X_train.shape[0], self.batch_size, replace=False)
    return self.X_train[sampled_classes], self.Y_train[sampled_classes]

@TinyJit
def jit_train(net, margin_loss, beta, X, Y, group_trainer):
  a_indices, anchors, positives, negatives, _ = net.sample(X)
  losses = margin_loss(Y, beta, a_indices, anchors, positives, negatives)
  group_trainer.zero_grad()
  losses.backward()
  group_trainer.step()
  return losses.realize()


def train(net, epochs, use_val=False):
  """Training function."""
    X_train, Y_train = fetch_cifar(train=True)
    X_test, Y_test = fetch_cifar(train=False)
    features = np.vstack([X_train, X_test])
    labels = np.hstack([Y_train, Y_test])
    self.X_train = features[labels < classes]
    self.Y_train = labels[labels < classes]
    self.X_test = features[labels >= classes]
    self.Y_test = labels[labels >= classes]
    del features
    del labels

  def gen_sample_train(self):
    # For CUB200, we use the first 100 classes for training.
    sampled_classes = np.random.choice(X_train.shape[0], opt.batch_size, replace=False)
    return X_train[sampled_classes], Y_train[sampled_classes]




  # train resnet separately th
  params_feature_detector = get_parameters(net.feature_net)
  params_embeddings = get_parameters(net.dense)

  group_trainer = OptimizerGroup()
  # dampen net -- todo only convs...
  group_trainer.append(optim.AdamW(params_feature_detector, lr=0.001 * 0.01, wd=0.00001, eps=1e-7))
  group_trainer.append(optim.AdamW(params_embeddings, lr=0.001, wd=0.00001, eps=1e-7))
  group_trainer.append(optim.SGD([beta], lr=0.1, momentum=0.9))

  margin_loss = MarginLoss(embed_dim=opt.embed_dim, batch_size=opt.batch_size, margin=opt.margin, nu=opt.nu, batch_k=opt.batch_k)
  
  best_val = 0.0

  hack = False

  for epoch in range(epochs):
    Tensor.training = True
    tic = time.time()
    prev_loss, cumulative_loss = 0.0, 0.0

    # Learning rate schedule.
    group_trainer[0].lr = get_lr(opt.lr, epoch, steps, opt.factor) 
    print(f'Epoch {epoch} learning rate={group_trainer[0].lr}')
    if opt.lr_beta > 0.0:
      group_trainer[-1].lr = get_lr(opt.lr_beta, epoch, steps, opt.factor)
      print(f'Epoch {epoch} beta learning rate={group_trainer[-1].lr}')
    
    # Inner training loop.
    for i in range(range_finder(opt.batch_size)):
      batch = train_data.next()
      data = batch.data[0]
      label = batch.label[0]

      # send to gpus/
      if isinstance(data, np.ndarray):
        data =  Tensor(data, requires_grad=False)
        label = Tensor(label, requires_grad=False)
    
      y_r = label.realize().numpy()

      losses = jit_train(net, margin_loss, beta, data, y_r, group_trainer)
      cumulative_loss = cumulative_loss + losses
      
      if (i+1) % opt.log_interval == 0:
      #if True:
        diff = cumulative_loss - prev_loss
        print(f'[Epoch {epoch}, Iter {i+1}] training loss={float(diff.numpy())}')
        prev_loss = cumulative_loss

    print(f'[Epoch {epoch}] training loss={float(cumulative_loss.numpy())}')
    print(f'[Epoch {epoch}] time cost: {time.time()-tic}')

    if use_val:
        names, val_accs = test_faiss(net, val_data, epoch,
                                        opt.save_dir, opt.save_model_prefix,
                                        plot=False)
        for name, val_acc in zip(names, val_accs):
            print(f'[Epoch {epoch}] validation: {name}={val_acc}')

        if val_accs[0] > best_val:
            best_val = val_accs[0]
            print(f'Saving {opt.save_model_prefix}.')
            # net.save('%s.params' % opt.save_model_prefix)
  return best_val