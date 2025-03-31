# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
from typing import Callable, Optional

import numpy as np
import warp as wp
from warp.context import Module
from warp.context import get_module

from . import types

_STACK = None


class EventTracer:
  """Calculates elapsed times of functions annotated with `event_scope`.

  Use as a context manager like so:

    @event_trace
    def my_warp_function(...):
      ...

    with EventTracer() as tracer:
      my_warp_function(...)
      print(tracer.trace())
  """

  def __init__(self, enabled: bool = True):
    global _STACK
    if _STACK is not None:
      raise ValueError("only one EventTracer can run at a time")
    if enabled:
      _STACK = {}

  def __enter__(self):
    return self

  def trace(self) -> dict:
    """Calculates elapsed times for every node of the trace."""
    global _STACK

    if _STACK is None:
      return {}

    ret = {}

    for k, v in _STACK.items():
      events, sub_stack = v
      # push into next level of stack
      saved_stack, _STACK = _STACK, sub_stack
      sub_trace = self.trace()
      # pop!
      _STACK = saved_stack
      events = tuple(wp.get_event_elapsed_time(beg, end) for beg, end in events)
      ret[k] = (events, sub_trace)

    return ret

  def __exit__(self, type, value, traceback):
    global _STACK
    _STACK = None


def _merge(a: dict, b: dict) -> dict:
  """Merges two event trace stacks."""
  ret = {}
  if not a or not b:
    return dict(**a, **b)
  if set(a) != set(b):
    raise ValueError("incompatible stacks")
  for key in a:
    a1_events, a1_substack = a[key]
    a2_events, a2_substack = b[key]
    ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
  return ret


def event_scope(fn, name: str = ""):
  """Wraps a function and records an event before and after the fucntion invocation."""
  name = name or getattr(fn, "__name__")

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    global _STACK
    if _STACK is None:
      return fn(*args, **kwargs)
    # push into next level of stack
    saved_stack, _STACK = _STACK, {}
    beg = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(beg)
    res = fn(*args, **kwargs)
    wp.record_event(end)
    # pop back up to current level
    sub_stack, _STACK = _STACK, saved_stack
    # append events and substack
    prev_events, prev_substack = _STACK.get(name, ((), {}))
    events = prev_events + ((beg, end),)
    sub_stack = _merge(prev_substack, sub_stack)
    _STACK[name] = (events, sub_stack)
    return res

  return wrapper


# @kernel decorator to automatically set up modules based on nested
# function names
def kernel(
  f: Optional[Callable] = None,
  *,
  enable_backward: Optional[bool] = None,
  module: Optional[Module] = None,
):
  """
  Decorator to register a Warp kernel from a Python function.
  The function must be defined with type annotations for all arguments.
  The function must not return anything.

  Example::

      @kernel
      def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
          tid = wp.tid()
          b[tid] = a[tid] + 1.0


      @kernel(enable_backward=False)
      def my_kernel_no_backward(a: wp.array(dtype=float, ndim=2), x: float):
          # the backward pass will not be generated
          i, j = wp.tid()
          a[i, j] = x


      @kernel(module="unique")
      def my_kernel_unique_module(a: wp.array(dtype=float), b: wp.array(dtype=float)):
          # the kernel will be registered in new unique module created just for this
          # kernel and its dependent functions and structs
          tid = wp.tid()
          b[tid] = a[tid] + 1.0

  Args:
      f: The function to be registered as a kernel.
      enable_backward: If False, the backward pass will not be generated.
      module: The :class:`warp.context.Module` to which the kernel belongs. Alternatively, if a string `"unique"` is provided, the kernel is assigned to a new module named after the kernel name and hash. If None, the module is inferred from the function's module.

  Returns:
      The registered kernel.
  """
  if module is None:
    # create a module name based on the name of the nested function
    # get the qualified name, e.g. "main.<locals>.nested_kernel"
    qualname = f.__qualname__
    parts = [part for part in qualname.split(".") if part != "<locals>"]
    outer_functions = parts[:-1]
    module = get_module(".".join([f.__module__] + outer_functions))

  return wp.kernel(f, enable_backward=enable_backward, module=module)


@wp.kernel
def _copy_2df(dest: types.array2df, src: types.array2df):
  i, j = wp.tid()
  dest[i, j] = src[i, j]


@wp.kernel
def _copy_3df(dest: types.array3df, src: types.array3df):
  i, j, k = wp.tid()
  dest[i, j, k] = src[i, j, k]


@wp.kernel
def _copy_2dvec10f(
  dest: wp.array2d(dtype=types.vec10f), src: wp.array2d(dtype=types.vec10f)
):
  i, j = wp.tid()
  dest[i, j] = src[i, j]


@wp.kernel
def _copy_2dvec3f(dest: wp.array2d(dtype=wp.vec3f), src: wp.array2d(dtype=wp.vec3f)):
  i, j = wp.tid()
  dest[i, j] = src[i, j]


@wp.kernel
def _copy_2dmat33f(dest: wp.array2d(dtype=wp.mat33f), src: wp.array2d(dtype=wp.mat33f)):
  i, j = wp.tid()
  dest[i, j] = src[i, j]


@wp.kernel
def _copy_2dspatialf(
  dest: wp.array2d(dtype=wp.spatial_vector), src: wp.array2d(dtype=wp.spatial_vector)
):
  i, j = wp.tid()
  dest[i, j] = src[i, j]


# TODO(team): remove kernel_copy once wp.copy is supported in cuda subgraphs


def kernel_copy(dest: wp.array, src: wp.array):
  if src.shape != dest.shape:
    raise ValueError("only same shape copying allowed")

  if src.dtype != dest.dtype:
    if (src.dtype, dest.dtype) not in (
      (wp.float32, np.float32),
      (np.float32, wp.float32),
      (wp.int32, np.int32),
      (np.int32, wp.int32),
    ):
      raise ValueError(f"only same dtype copying allowed: {src.dtype} != {dest.dtype}")

  if src.ndim == 2 and src.dtype == wp.float32:
    kernel = _copy_2df
  elif src.ndim == 3 and src.dtype == wp.float32:
    kernel = _copy_3df
  elif src.ndim == 2 and src.dtype == wp.vec3f:
    kernel = _copy_2dvec3f
  elif src.ndim == 2 and src.dtype == wp.mat33f:
    kernel = _copy_2dmat33f
  elif src.ndim == 2 and src.dtype == types.vec10f:
    kernel = _copy_2dvec10f
  elif src.ndim == 2 and src.dtype == wp.spatial_vector:
    kernel = _copy_2dspatialf
  else:
    raise NotImplementedError("copy not supported for these array types")

  wp.launch(kernel=kernel, dim=src.shape, inputs=[dest, src])
