
from __future__ import division

import argparse
import logging
import time

import numpy as np

from data import cub200_iterator

from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from model import MarginNet, DistanceWeightedMarginLoss
from resnet import ResNet50
from tinygrad.state import get_parameters
from tinygrad.jit import TinyJit

from evaluation import evaluate_emb_faiss, test_faiss

# logging.basicConfig(level=logging.INFO)

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
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.2,
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

# logging.info(opt)


class OptimizerGroup(list):
  def zero_grad(self): (optimizer.zero_grad() for optimizer in self)
  def step(self): (optimizer.step() for optimizer in self)
       


def get_lr(lr, epoch, steps, factor):
  """Get learning rate based on schedule."""
  for s in steps:
    if epoch >= s:
      lr *= factor
  return lr

#TODO actually get this to work with the random function
@TinyJit
def train_step_jitted(net, margin_loss, beta, X, Y, group_trainer):
  embeddings = net(X)
  #embeddings = infrence_jitted(net, data) #Nothing to Jit..?
  losses = margin_loss.get(embeddings, Y, beta)
  losses = losses[0] if isinstance(losses, tuple) else losses
  group_trainer.zero_grad()

  losses.backward()

  group_trainer.step()

  return losses.realize()

@TinyJit
def infrence_jitted(net, X):
  # doesn't work when training=True...
  embeddings = net(X)
  return embeddings.realize()

def range_finder(batch_size): return 14000 // batch_size

def train(net, epochs, use_val=False):
  """Training function."""
  
  # train resnet separately th
  
  params_feature_detector = get_parameters(net.feature_detector)
  params_embeddings = get_parameters(net.dense)
  
  group_trainer = OptimizerGroup()
  # dampen net
  group_trainer.append(optim.AdamW(params_feature_detector, lr= opt.lr * 0.01, wd=opt.wd, eps=1e-7))
  group_trainer.append(optim.AdamW(params_embeddings, lr= opt.lr, wd=opt.wd, eps=1e-7))
  if opt.lr_beta > 0.0:
    group_trainer.append(optim.SGD([beta], lr = opt.lr_beta, momentum = 0.9))

  margin_loss = DistanceWeightedMarginLoss(embed_dim=opt.embed_dim, batch_size=opt.batch_size, margin=opt.margin, nu=opt.nu, batch_k=opt.batch_k)
  Tensor.training = True
  
  best_val = 0.0

  hack = False

  for epoch in range(epochs):
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
    
      #embeddings = net(data)
      embeddings = infrence_jitted(net, data) #Nothing to Jit..?
      losses = margin_loss.get(embeddings, label, beta)
      losses = losses[0] if isinstance(losses, tuple) else losses

      group_trainer.zero_grad()
      # compute gradient and do SGD steps
      losses.backward()
      group_trainer.step()
      # subprocess.call('nvidia-smi')
      '''
      if not hack:
        losses = train_step_jitted(net, margin_loss, beta, data, label, group_trainer)
        train_step_jitted.cnt = 3
        hack = True
      else:
        losses = train_step_jitted(net, margin_loss, beta, data, label, group_trainer)
      '''
      cumulative_loss = cumulative_loss + losses.realize()
      
      if (i+1) % opt.log_interval == 0:
        diff = cumulative_loss - prev_loss
        print(f'[Epoch {epoch}, Iter {i+1}] training loss={float(diff.numpy())}')
        prev_loss = cumulative_loss

    print(f'[Epoch {epoch}] training loss={float(cumulative_loss.numpy())}')
    print(f'[Epoch {epoch}] time cost: {time.time()-tic}')

    if use_val:
        names, val_accs = test_faiss(net, val_data, epoch,
                                        opt.save_dir, opt.save_model_prefix)
        for name, val_acc in zip(names, val_accs):
            print(f'[Epoch {epoch}] validation: {name}={val_acc}')

        if val_accs[0] > best_val:
            best_val = val_accs[0]
            print(f'Saving {opt.save_model_prefix}.')
            # net.save('%s.params' % opt.save_model_prefix)
  return best_val


if __name__ == '__main__':

  # Settings.
  np.random.seed(opt.seed)

  batch_size = opt.batch_size

  steps = [int(step) for step in opt.steps.split(',')]

  net = MarginNet(opt.embed_dim, opt.batch_size)
  net.load_basenet_features() # load the basenet
  beta = Tensor(np.ones((100,)).astype(np.float32) * opt.beta)

  # Get iterators.
  train_data, val_data = cub200_iterator(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))




  best_val_recall = train(net, opt.epochs)
  print('Best validation Recall@1: %.2f.' % best_val_recall)


  def deepwalk(self):
    def _deepwalk(node, visited, nodes, fails):
      visited.add(node)
      try:
        if node._ctx:
          for i in node._ctx.parents:
            if i not in visited: _deepwalk(i, visited, nodes)
          nodes.append(node)
        return nodes
      except:
        print('fail!')
        fails.append(node)
        return node
    
    return _deepwalk(self, set(), [], [])


  # ***** toposort and backward pass *****
  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx:
        for i in node._ctx.parents:
          if i not in visited: _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])


    def _deepwalk(node, visited, nodes):
      visited.add(node)
      try:
        if node._ctx:
          for i in node._ctx.parents:
            if i not in visited: _deepwalk(i, visited, nodes)
          nodes.append(node)
        return nodes
      except:
        print('fail')
        fails.append(node)
        return node