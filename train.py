
from __future__ import division

import argparse
import logging
import time

import numpy as np
from bottleneck import argpartition

from data import cub200_iterator

from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from model import MarginNet, MarginLoss, ResNetFeats, get_distance, DistanceWeightedMarginLoss
#from ..resnet import ResNet50
from tinygrad.state import get_parameters
from tinygrad.jit import TinyJit

from evaluation import evaluate_emb_faiss, test_faiss

logging.basicConfig(level=logging.INFO)

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

logging.info(opt)

def get_lr(lr, epoch, steps, factor):
  """Get learning rate based on schedule."""
  for s in steps:
    if epoch >= s:
      lr *= factor
  return lr

#TODO actually get this to work with the random function
@TinyJit
def train_step_jitted(net, margin_loss, beta, X, Y, trainer, trainer_beta):
  embeddings = net(X)
  losses, pair_cnt = margin_loss(embeddings, Y, beta)
  trainer.zero_grad()
  trainer_beta.zero_grad()
  losses.backward()
  trainer.step()
  trainer_beta.step()
  return losses.realize()

@TinyJit
def infrence_jitted(net, X):
  embeddings = net(X)
  return embeddings

def train(net, epochs):
  """Training function."""
  parameters = get_parameters(net)
  trainer = optim.AdamW(parameters, lr=opt.lr, wd=opt.wd, eps=1e-7)

  if opt.lr_beta > 0.0:
    trainer_beta = optim.SGD([beta], lr = opt.lr_beta, momentum = 0.9)

  margin_loss = DistanceWeightedMarginLoss(batch_size=opt.batch_size, margin=opt.margin, nu=opt.nu, batch_k=opt.batch_k)
  Tensor.training = True

  best_val = 0.0
  for epoch in range(epochs):
    tic = time.time()
    prev_loss, cumulative_loss = 0.0, 0.0

    # Learning rate schedule.
    trainer.lr = get_lr(opt.lr, epoch, steps, opt.factor)
    logging.info('Epoch %d learning rate=%f', epoch, trainer.lr)
    if opt.lr_beta > 0.0:
      trainer_beta.lr = get_lr(opt.lr_beta, epoch, steps, opt.factor)
      logging.info('Epoch %d beta learning rate=%f', epoch, trainer_beta.lr)
    
    # Inner training loop.
    for i in range(200):
      batch = train_data.next()
      data = batch.data[0]
      label = batch.label[0]

      embeddings = net(data)
      losses, pair_cnt = margin_loss(embeddings, label, beta)

      # compute gradient and do SGD step``
      trainer.zero_grad()
      trainer_beta.zero_grad()
      losses.backward()
      trainer.step()
      trainer_beta.step()

      cumulative_loss = losses
      
      if (i+1) % opt.log_interval == 0:
        diff = cumulative_loss - prev_loss
        logging.info('[Epoch %d, Iter %d] training loss=%f' % (
          epoch, i+1, float(diff.numpy())))
        prev_loss = cumulative_loss

    logging.info('[Epoch %d] training loss=%f'%(epoch, float(cumulative_loss.numpy())))
    logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

    names, val_accs = test_faiss(net, val_data, epoch,
                                     opt.save_dir, opt.save_model_prefix)
    for name, val_acc in zip(names, val_accs):
      logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))

    if val_accs[0] > best_val:
      best_val = val_accs[0]
      logging.info('Saving %s.' % opt.save_model_prefix)
      # net.save('%s.params' % opt.save_model_prefix)
  return best_val


if __name__ == '__main__':

  # Settings.
  np.random.seed(opt.seed)

  batch_size = opt.batch_size

  steps = [int(step) for step in opt.steps.split(',')]

  # Construct model.
  net = ResNet50()
  net.load_from_pretrained()
  features = ResNetFeats(net, batch_size=opt.batch_size)

  # not sure how to do this...
  #if opt.use_pretrained:
  #  # Use a smaller learning rate for pre-trained convolutional layers.
  #  for v in net.collect_params().values():
  #    if 'conv' in v.name:
  #      setattr(v, 'lr_mult', 0.01)

  net = MarginNet(features, 2048, opt.embed_dim, opt.batch_k)
  beta = Tensor.ones(100,)

  # Get iterators.
  train_data, val_data = cub200_iterator(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))
  best_val_recall = train(net, opt.epochs)
  print('Best validation Recall@1: %.2f.' % best_val_recall)


'''
I also got this problem and solve it, maybe it will help someone:

In my situation (two versions of swig in the system, in /usr/bin and installed by me manually into the conda env) this error rises only when just cmake flags are stated, but $PATH not updated. So to solve this one need to:

add this cmake flags to cmake command:
-DSWIG_DIR=~/miniconda3/envs/crpropa/share/swig/4.0.2 - this is the result of command 'swig -swiglib'
-DSWIG_EXECUTABLE=~/miniconda3/envs/crpropa/bin/swig
AND add the path to the correct version of swig into the beginning of the $PATH variable:
export PATH=~/miniconda3/envs/crpropa/bin:$PATH

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[6], line 239
    235   return best_val
    238 if __name__ == '__main__':
--> 239   best_val_recall = train(net, opt.epochs)
    240   print('Best validation Recall@1: %.2f.' % best_val_recall)
    243 
    244 I also got this problem and solve it, maybe it will help someone:
    245 
   (...)
    252 export PATH=~/miniconda3/envs/crpropa/bin:$PATH
    253 

Cell In[6], line 234, in train(net, epochs)
    232     best_val = val_accs[0]
    233     logging.info('Saving %s.' % opt.save_model_prefix)
--> 234     net.save('%s.params' % opt.save_model_prefix)
    235 return best_val

File ~/Downloads/compressed/model.py:108, in MarginNet.save(self, filename)
    105 with open(filename+'.npy', 'wb') as f:
    106   for par in get_parameters(self):
    107     #if par.requires_grad:
--> 108     np.save(f, par.cpu().numpy())

File ~/github/tinygrad/tinygrad/tensor.py:111, in Tensor.numpy(self)
--> 111 def numpy(self) -> np.ndarray: return self.lazydata.toCPU()

File ~/github/tinygrad/tinygrad/lazy.py:158, in LazyBuffer.toCPU(self)
    157 def toCPU(self):
--> 158   realized = self.cast(dtypes.from_np(self.dtype.np)).contiguous().realize().realized
    159   ret = cast(RawBuffer, realized).toCPU().reshape(self.shape)
    160   return ret

File ~/github/tinygrad/tinygrad/lazy.py:114, in LazyBuffer.realize(self)
    112 if self.optype is BinaryOps: self.op = _ast_binaryops(self)
    113 elif self.optype is ReduceOps: self.op = _ast_reduceops(self)
--> 114 elif self.optype is LoadOps: LOAD_OPS_DISPATCHER[cast(LoadOps, self.op.op)](self)
    115 # run the ast if we still have to, and log the op
    116 if not self.realized:

File ~/github/tinygrad/tinygrad/lazy.py:306, in _realize_contiguous(buffer)
    305 def _realize_contiguous(buffer: LazyBuffer) -> None:
--> 306   realized = buffer.op.src[0].realize().realized
    307   if buffer.op.src[0].st.contiguous and realized.__class__ is not RawConst and cast(RawBuffer, realized).size == prod(buffer.shape):
    308     # no need to run an AST, this is already contiguous
    309     buffer.realized = realized

File ~/github/tinygrad/tinygrad/lazy.py:114, in LazyBuffer.realize(self)
    112 if self.optype is BinaryOps: self.op = _ast_binaryops(self)
    113 elif self.optype is ReduceOps: self.op = _ast_reduceops(self)
--> 114 elif self.optype is LoadOps: LOAD_OPS_DISPATCHER[cast(LoadOps, self.op.op)](self)
    115 # run the ast if we still have to, and log the op
    116 if not self.realized:

File ~/github/tinygrad/tinygrad/lazy.py:319, in _realize_from(buffer)
    318 def _realize_from(buffer: LazyBuffer) -> None:
--> 319   rawbuf = buffer.op.src[0].realize()
    320   # TODO: make this generic
    321   if isinstance(rawbuf.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):

File ~/github/tinygrad/tinygrad/lazy.py:127, in LazyBuffer.realize(self)
    125       self.op = LazyOp(UnaryOps.CAST, (self.op,), dtypes.float32)
    126     self.dtype = dtypes.float32
--> 127   self.realized = Device[self.device].exec_ast(self.op, output=self, **self._device_extra_args())
    129 assert self.realized and isinstance(self.realized, (RawConst, Device[self.device].buffer)), f"device mismatch on realized got {type(self.realized)} expected {self.device}"
    130 # HACK: allow hot casting of images

File ~/github/tinygrad/tinygrad/ops.py:183, in Compiled.exec_ast(self, ast, output, **kwargs)
    181 # this is the default now
    182 if hasattr(k, 'key') and getenv("ENABLE_METHOD_CACHE", 1):
--> 183   if k.key not in self.method_cache: self.method_cache[k.key] = k.codegen().build(self.runtime)
    184   elif DEBUG >= 5: print(f"method cache hit : {k.key}")
    185   prg = self.method_cache[k.key]

File ~/github/tinygrad/tinygrad/ops.py:133, in ASTRunner.build(self, runtime)
    132 def build(self, runtime):
--> 133   self.clprg = runtime(self.name, self.prg, **self.runtime_args)
    134   return self

File ~/github/tinygrad/tinygrad/runtime/ops_gpu.py:59, in CLProgram.__init__(self, name, prg, binary, argdtypes, options)
     57 except cl.RuntimeError as e:
     58   if DEBUG >= 3: print("FAILED TO BUILD", prg)
---> 59   raise e
     60 self.clprg = self._clprg.__getattr__(name)
     61 if DEBUG >= 5 and not OSX:

File ~/github/tinygrad/tinygrad/runtime/ops_gpu.py:56, in CLProgram.__init__(self, name, prg, binary, argdtypes, options)
     54 self.name, self.argdtypes, self.clprogram = name, argdtypes, cl.Program(CL.cl_ctx, CL.cl_ctx.devices, [prg]*len(CL.cl_ctx.devices)) if binary else cl.Program(CL.cl_ctx, prg)  # type: ignore
     55 try:
---> 56   self._clprg = self.clprogram.build(options=options)
     57 except cl.RuntimeError as e:
     58   if DEBUG >= 3: print("FAILED TO BUILD", prg)

File ~/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/__init__.py:534, in Program.build(self, options, devices, cache_dir)
    530 else:
    531     # cached
    533     from pyopencl.cache import create_built_program_from_source_cached
--> 534     self._prg, was_cached = self._build_and_catch_errors(
    535             lambda: create_built_program_from_source_cached(
    536                 self._context, self._source, options_bytes, devices,
    537                 cache_dir=cache_dir, include_path=include_path),
    538             options_bytes=options_bytes, source=self._source)
    540     if was_cached:
    541         build_descr = "cache retrieval"

File ~/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/__init__.py:582, in Program._build_and_catch_errors(self, build_func, options_bytes, source)
    574     err = _cl.RuntimeError(
    575             _cl._ErrorRecord(
    576                 msg=msg,
    577                 code=code,
    578                 routine=routine))
    580 # Python 3.2 outputs the whole list of currently active exceptions
    581 # This serves to remove one (redundant) level from that nesting.
--> 582 raise err

RuntimeError: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'NVIDIA GeForce RTX 3070' on 'NVIDIA CUDA' at 0x2d773c0>:


(options: -I /home/mkrzus/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/cl)
(source saved as /tmp/tmphqf9j2dq.cl


AssertionError                            Traceback (most recent call last)
Cell In[6], line 244
    240   return best_val
    243 if __name__ == '__main__':
--> 244   best_val_recall = train(net, opt.epochs)
    245   print('Best validation Recall@1: %.2f.' % best_val_recall)

Cell In[6], line 207, in train(net, epochs)
    204 label = batch.label[0]
    206 # embeddings = net(data)
--> 207 embeddings = infrence_jitted(net, data)
    208 losses, pair_cnt = margin_loss(embeddings, label, beta)
    210 # compute gradient and do SGD step``

File ~/github/tinygrad/tinygrad/jit.py:37, in TinyJit.__call__(self, *args, **kwargs)
     35 self.jit_cache = GlobalCounters.cache
     36 GlobalCounters.cache = None
---> 37 assert len(self.jit_cache) != 0, "didn't JIT anything!"
     38 if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")
     40 # get the inputs for replacement

AssertionError: didn't JIT anything!

RuntimeError                              Traceback (most recent call last)
Cell In[6], line 187
    185   # Get iterators.
    186   train_data, val_data = cub200_iterator(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))
--> 187   best_val_recall = train(net, opt.epochs)
    188   print('Best validation Recall@1: %.2f.' % best_val_recall)
    191 '''
    192 I also got this problem and solve it, maybe it will help someone:
    193 
   (...)
    349 AssertionError: didn't JIT anything!
    350 '''

Cell In[6], line 134, in train(net, epochs)
    132 trainer_beta.zero_grad()
    133 losses.backward()
--> 134 trainer.step()
    135 trainer_beta.step()
    137 cumulative_loss = losses

File ~/github/tinygrad/tinygrad/nn/optim.py:69, in LAMB.step(self)
     67     r = 1.0
     68   t.assign(t.detach() - self.lr * r * up)
---> 69 self.realize([self.t] + self.m + self.v)

File ~/github/tinygrad/tinygrad/nn/optim.py:22, in Optimizer.realize(self, extra)
     18 def realize(self, extra=None):
     19   # TODO: corealize
     20   # NOTE: in extra is too late for most of the params due to issues with assign
     21   for p in extra + self.params + self.buffers if extra is not None else self.params + self.buffers:
---> 22     p.realize()

File ~/github/tinygrad/tinygrad/tensor.py:93, in Tensor.realize(self)
     92 def realize(self) -> Tensor:
---> 93   self.lazydata.realize()
     94   return self

File ~/github/tinygrad/tinygrad/lazy.py:127, in LazyBuffer.realize(self)
    125       self.op = LazyOp(UnaryOps.CAST, (self.op,), dtypes.float32)
    126     self.dtype = dtypes.float32
--> 127   self.realized = Device[self.device].exec_ast(self.op, output=self, **self._device_extra_args())
    129 assert self.realized and isinstance(self.realized, (RawConst, Device[self.device].buffer)), f"device mismatch on realized got {type(self.realized)} expected {self.device}"
    130 # HACK: allow hot casting of images

File ~/github/tinygrad/tinygrad/ops.py:183, in Compiled.exec_ast(self, ast, output, **kwargs)
    181 # this is the default now
    182 if hasattr(k, 'key') and getenv("ENABLE_METHOD_CACHE", 1):
--> 183   if k.key not in self.method_cache: self.method_cache[k.key] = k.codegen().build(self.runtime)
    184   elif DEBUG >= 5: print(f"method cache hit : {k.key}")
    185   prg = self.method_cache[k.key]

File ~/github/tinygrad/tinygrad/ops.py:133, in ASTRunner.build(self, runtime)
    132 def build(self, runtime):
--> 133   self.clprg = runtime(self.name, self.prg, **self.runtime_args)
    134   return self

File ~/github/tinygrad/tinygrad/runtime/ops_gpu.py:59, in CLProgram.__init__(self, name, prg, binary, argdtypes, options)
     57 except cl.RuntimeError as e:
     58   if DEBUG >= 3: print("FAILED TO BUILD", prg)
---> 59   raise e
     60 self.clprg = self._clprg.__getattr__(name)
     61 if DEBUG >= 5 and not OSX:

File ~/github/tinygrad/tinygrad/runtime/ops_gpu.py:56, in CLProgram.__init__(self, name, prg, binary, argdtypes, options)
     54 self.name, self.argdtypes, self.clprogram = name, argdtypes, cl.Program(CL.cl_ctx, CL.cl_ctx.devices, [prg]*len(CL.cl_ctx.devices)) if binary else cl.Program(CL.cl_ctx, prg)  # type: ignore
     55 try:
---> 56   self._clprg = self.clprogram.build(options=options)
     57 except cl.RuntimeError as e:
     58   if DEBUG >= 3: print("FAILED TO BUILD", prg)

File ~/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/__init__.py:534, in Program.build(self, options, devices, cache_dir)
    530 else:
    531     # cached
    533     from pyopencl.cache import create_built_program_from_source_cached
--> 534     self._prg, was_cached = self._build_and_catch_errors(
    535             lambda: create_built_program_from_source_cached(
    536                 self._context, self._source, options_bytes, devices,
    537                 cache_dir=cache_dir, include_path=include_path),
    538             options_bytes=options_bytes, source=self._source)
    540     if was_cached:
    541         build_descr = "cache retrieval"

File ~/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/__init__.py:582, in Program._build_and_catch_errors(self, build_func, options_bytes, source)
    574     err = _cl.RuntimeError(
    575             _cl._ErrorRecord(
    576                 msg=msg,
    577                 code=code,
    578                 routine=routine))
    580 # Python 3.2 outputs the whole list of currently active exceptions
    581 # This serves to remove one (redundant) level from that nesting.
--> 582 raise err

RuntimeError: clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE - clBuildProgram failed: BUILD_PROGRAM_FAILURE

Build on <pyopencl.Device 'NVIDIA GeForce RTX 3070' on 'NVIDIA CUDA' at 0x1681910>:


(options: -I /home/mkrzus/anaconda3/envs/lightning/lib/python3.10/site-packages/pyopencl/cl)
(source saved as /tmp/tmp5dp67i3l.cl)



'''