
from __future__ import division
from typing import List
import argparse
import logging
import time

import numpy as np

from data import cub200_iterator, cub200_iterator2

from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from model import MarginNet, MarginLoss
from tinygrad.state import get_parameters
from tinygrad.jit import TinyJit

from evaluation import evaluate_emb_faiss, test_faiss

# CLI
parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--data-path', type=str, default='cub200_data/CUB_200_2011',
                    help='path of data.')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--batch-size', type=int, default=10,
                    help='training batch size per device (CPU/GPU). default is 70.')
parser.add_argument('--batch-k', type=int, default=5,
                    help='number of images per class in a batch. default is 5.')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to use, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs. default is 20.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer. default is adam.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.1,
                    help='learning rate for the beta in margin based loss. default is 0.1.')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for the margin based loss. default is 0.2.')
parser.add_argument('--beta', type=float, default=1.2,
                    help='initial value for beta. default is 1.2.')
parser.add_argument('--nu', type=float, default=0.0,
                    help='regularization parameter for beta. default is 0.0.')
parser.add_argument('--factor', type=float, default=0.5,
                    help='learning rate schedule factor. default is 0.5.')
parser.add_argument('--steps', type=str, default='12,14,16,18',
                    help='epochs to update learning rate. default is 12,14,16,18.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. default=123.')
parser.add_argument('--model', type=str, default='resnet50_v2',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--save-model-prefix', type=str, default='margin_loss_model',
                    help='prefix of models to be saved.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging.')
parser.add_argument('--save-dir', type=str, default='.',
                    help='the dir to save to')                    
opt = parser.parse_args()

# logging.info(opt) weird jit issue... it wouldn't 
# take this form?
class OptimizerGroup(list):
  def zero_grad(self): [optimizer.zero_grad() for optimizer in self]
  def step(self): [optimizer.step() for optimizer in self]
       
class OptimizerGroup2: 
  def __init__(self, optimizers): self.optimizers = optimizers
  def zero_grad(self): (optimizer.zero_grad() for optimizer in self.optimizers)
  def step(self): (optimizer.step() for optimizer in self.optimizers)
  
def zero_grad(optimizers): (optimizer.zero_grad() for optimizer in optimizers)
def step(optimizers): (optimizer.step() for optimizer in optimizers)

def get_lr(lr, epoch, steps, factor):
  """Get learning rate based on schedule."""
  for s in steps:
    if epoch >= s:
      lr *= factor
  return lr

@TinyJit
def jit_train_old(net, margin_loss, beta, X, Y, group_trainer):
  a_indices, p_indices, n_indices, embeddings = net.sample(X)
  losses = margin_loss(embeddings, Y, beta, a_indices, p_indices, n_indices)
  group_trainer.zero_grad()
  losses.backward()
  group_trainer.step()
  return losses.realize()

@TinyJit
def jit_train(net, margin_loss, beta, X, Y, group_trainer):
  batch_size = X.shape[0]
  a_indices, anchors, positives, negatives, embeddings = net.sample(X)
  losses = margin_loss(Y, beta, a_indices, anchors, positives, negatives)
  if isinstance(group_trainer, OptimizerGroup):
    group_trainer.zero_grad()
  else:
    for g in group_trainer:
      g.zero_grad()
  losses.backward()
  if isinstance(group_trainer, OptimizerGroup):
    group_trainer.step()
  else:
    for g in group_trainer:
      g.step(batch_size)
  return losses.realize()

@TinyJit
def infrence_jitted(net, X):
  embeddings = net(X)
  return embeddings.realize().numpy()

def test_faiss(net, val_data, epoch, save_dir='', save_model_prefix='', plot=True):
  """Test a model."""
  print('starting test')
  Tensor.training = False
  val_data.reset()
  val_data.reset()
  outputs = []
  labels = []
  count = 0
  for batch in val_data:
    data = batch.data[0]
    label = batch.label[0]
    if data.__class__ is np.ndarray:
      data =  Tensor(data, requires_grad=False)
    outputs.append(infrence_jitted(net, data))
    labels.append(label)
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
  return evaluate_emb_faiss(outputs, labels, val_data, save_dir, save_model_prefix, epoch=epoch, plot=plot)

def range_finder(batch_size): return 14000 // batch_size

from tinygrad.nn.optim import Optimizer
# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW2(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01): return LAMB2(params, lr, b1, b2, eps, wd, adam=True)
def Adam2(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8): return LAMB2(params, lr, b1, b2, eps, 0.0, adam=True)

class LAMB2(Optimizer):
  def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd, self.adam, self.t = b1, b2, eps, wd, adam, Tensor([0], requires_grad=False).realize()
    self.m = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]

  def step(self, batch_size=1) -> None:
    rescale_grad = 1 / batch_size
    self.t.assign(self.t + 1).realize()
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = (t.grad.realize() * rescale_grad) if rescale_grad != 1 else t.grad.realize() # no mult op
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g).realize()
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).realize()
      m_hat = self.m[i] / (1.0 - self.b1**self.t)
      v_hat = self.v[i] / (1.0 - self.b2**self.t)
      up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      t.assign(t.detach() - self.lr * r * up)
    self.realize([self.t] + self.m + self.v)

def train(net, epochs, use_val=False):
  """Training function."""
  # train resnet separately th
  params_feature_detector = get_parameters(net.feature_net)
  params_embeddings = get_parameters(net.dense)
  use_optimizer_group = True
  group_trainer = OptimizerGroup() if use_optimizer_group else []
  # dampen net -- todo only convs...
  group_trainer.append(AdamW2(params_feature_detector, lr= opt.lr * 0.01, wd=opt.wd, eps=1e-7))
  group_trainer.append(AdamW2(params_embeddings, lr= opt.lr, wd=opt.wd, eps=1e-7))
  if opt.lr_beta > 0.0:
    group_trainer.append(AdamW2([beta], lr = opt.lr_beta))

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


def debug1(net, epochs, use_val=False):
  """Training function."""
  # train resnet separately th
  params_feature_detector = get_parameters(net.feature_net)
  params_embeddings = get_parameters(net.dense)
  use_optimizer_group = True
  group_trainer = OptimizerGroup() if use_optimizer_group else []
  # dampen net -- todo only convs...
  group_trainer.append(AdamW2(params_feature_detector, lr= opt.lr * 0.01, wd=opt.wd, eps=1e-7))
  group_trainer.append(AdamW2(params_embeddings, lr= opt.lr, wd=opt.wd, eps=1e-7))
  if opt.lr_beta > 0.0:
    group_trainer.append(AdamW2([beta], lr = opt.lr_beta))

  margin_loss = MarginLoss(embed_dim=opt.embed_dim, batch_size=opt.batch_size, margin=opt.margin, nu=opt.nu, batch_k=opt.batch_k)
  
  Tensor.training = True
  losses = []
  # Inner training loop.
  for i in range(3):
    batch = train_data.next()
    data = batch.data[0]
    label = batch.label[0]

    # send to gpus/
    if isinstance(data, np.ndarray):
      data =  Tensor(data, requires_grad=False)
      label = Tensor(label, requires_grad=False)
  
    y_r = label.realize().numpy()

    loss = jit_train(net, margin_loss, beta, data, y_r, group_trainer)
    # cumulative_loss = cumulative_loss + losses
    losses.append(loss.numpy())
  return losses

if __name__ == '__main__':

  # Settings.
  np.random.seed(opt.seed)

  batch_size = opt.batch_size

  steps = [int(step) for step in opt.steps.split(',')]

  net = MarginNet(opt.embed_dim, opt.batch_size, opt.batch_k)
  net.load_basenet_features() # load the basenet
  beta = Tensor(np.ones((100,)).astype(np.float32) * opt.beta)

  # Get iterators.
  train_data, val_data = cub200_iterator2(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))

  best_val_recall = train(net, opt.epochs, True)
  #print('Best validation Recall@1: %.2f.' % best_val_recall)

  #losses = debug1(net, opt.epochs)

  #print(losses)