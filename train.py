
from __future__ import division

import argparse
import logging
import time

import numpy as np

from data import cub200_iterator

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
  a_indices, anchors, positives, negatives, embeddings = net.sample(X)
  losses = margin_loss(Y, beta, a_indices, anchors, positives, negatives)
  group_trainer.zero_grad()
  losses.backward()
  group_trainer.step()
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

def train(net, epochs, use_val=False):
  """Training function."""
  # train resnet separately th
  params_feature_detector = get_parameters(net.feature_net)
  params_embeddings = get_parameters(net.dense)
  use_optimizer_group = True
  if use_optimizer_group:
    group_trainer = OptimizerGroup()
    # dampen net -- todo only convs...
    group_trainer.append(optim.AdamW(params_feature_detector, lr= opt.lr * 0.01, wd=opt.wd, eps=1e-7))
    group_trainer.append(optim.AdamW(params_embeddings, lr= opt.lr, wd=opt.wd, eps=1e-7))
    if opt.lr_beta > 0.0:
        group_trainer.append(optim.SGD([beta], lr = opt.lr_beta, momentum = 0.9))
  else:
    group_trainer = []
    # dampen net
    group_trainer.append(optim.AdamW(params_feature_detector, lr= opt.lr * 0.01, wd=opt.wd, eps=1e-7))
    group_trainer.append(optim.AdamW(params_embeddings, lr= opt.lr, wd=opt.wd, eps=1e-7))
    if opt.lr_beta > 0.0:
        group_trainer.append(optim.SGD([beta], lr = opt.lr_beta, momentum = 0.9))

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


if __name__ == '__main__':

  # Settings.
  np.random.seed(opt.seed)

  batch_size = opt.batch_size

  steps = [int(step) for step in opt.steps.split(',')]

  net = MarginNet(opt.embed_dim, opt.batch_size, opt.batch_k)
  net.load_basenet_features() # load the basenet
  beta = Tensor(np.ones((100,)).astype(np.float32) * opt.beta)

  # Get iterators.
  train_data, val_data = cub200_iterator(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))

  best_val_recall = train(net, opt.epochs, True)
  print('Best validation Recall@1: %.2f.' % best_val_recall)






def tests(net1, net2, train_inter)



















































class LargeJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Callable, Any]] = []  # TODO: Any should be List[RawBuffer], but this fails
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], int, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_size, expected_type)

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    self.aargs = [x for x in args]
    self.kwargs = [x for x in kwargs]
    if Device.DEFAULT not in ["GPU", "CLANG", "METAL", "CUDA", "HIP"]: return self.fxn(*args, **kwargs)  # only jit on the GPU codegen
    # NOTE: this cast is needed since although we know realize will create a ".realized" DeviceBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {cast(Union[int, str], k):cast(RawBuffer, v.realize().lazydata.realized) for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    self.og_input_rawbuffers = input_rawbuffers
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      for (j,i),(input_name, expected_size, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name].size == expected_size and input_rawbuffers[input_name].dtype == expected_type, f"size or type mismatch in JIT, {input_rawbuffers[input_name]} != <{expected_size}, {expected_type}>"
        self.jit_cache[j][1][i] = input_rawbuffers[input_name]
      for prg, args in self.jit_cache: prg(args, jit=True)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,(prg,args) in enumerate(self.jit_cache):  # pylint: disable=E1133
        for i,a in enumerate(args):
          if a in input_rawbuffers.values():
            self.input_replace[(j,i)] = [(k, v.size, v.dtype) for k,v in input_rawbuffers.items() if v == a][0]
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      #GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      #self.jit_cache = GlobalCounters.cache
      #GlobalCounters.cache = None
    self.cnt += 1
    return self.ret

@LargeJit
def train_jit(net, margin_loss, beta, X, Y, group_trainer):
  embeddings = net(X)
  #embeddings = infrence_jitted(net, data) #Nothing to Jit..?
  losses = margin_loss(embeddings, Y, beta)
  losses = losses[0] if isinstance(losses, tuple) else losses
  group_trainer.zero_grad()
  # compute gradient and do SGD steps
  #trainer.zero_grad()
  #trainer_beta.zero_grad()
  losses.backward()
  #trainer.step()
  #trainer_beta.step()
  group_trainer.step()
  # subprocess.call('nvidia-smi')
  return losses.realize()

    input_rawbuffers: Dict[Union[int, str], RawBuffer] = {cast(Union[int, str], k):cast(RawBuffer, v.realize().lazydata.realized) for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      for (j,i),(input_name, expected_size, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name].size == expected_size and input_rawbuffers[input_name].dtype == expected_type, f"size or type mismatch in JIT, {input_rawbuffers[input_name]} != <{expected_size}, {expected_type}>"
        self.jit_cache[j][1][i] = input_rawbuffers[input_name]
      for prg, args in self.jit_cache: prg(args, jit=True)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*self.aargs, {})
      self.jit_cache = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,(prg,args) in enumerate(self.jit_cache):  # pylint: disable=E1133
        for i,a in enumerate(args):
          if a in input_rawbuffers.values():
            self.input_replace[(j,i)] = [(k, v.size, v.dtype) for k,v in input_rawbuffers.items() if v == a][0]
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret


#In [55]: input_rawbuffers
Out[55]: 
{2: buffer<100, dtypes.float>,
 3: buffer<1505280, dtypes.float>,
 4: buffer<10, dtypes.long>}

In [56]: input_rawbuffers[4]
Out[56]: buffer<10, dtypes.long>

In [57]: input_rawbuffers[4]



'''
input_rawbuffers[4] is not in jit_cache during the first pass. 

'''
# get the inputs for replacement
for j,(prg,args) in enumerate(self.jit_cache):  # pylint: disable=E1133
  for i,a in enumerate(args):
    if a in input_rawbuffers.values():
      self.input_replace[(j,i)] = [(k, v.size, v.dtype) for k,v in input_rawbuffers.items() if v == a][0]
    else:
      self.Input_replace[(j,i)] = a


'''
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
'''