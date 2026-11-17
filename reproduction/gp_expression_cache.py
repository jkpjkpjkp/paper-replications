"""Bounded, opt-in memoization for immutable upstream GP expression evaluation.

This changes no RNG, operator arithmetic, fitness, or evolution settings. Cache
entries own detached tensor clones and hits return clones, so callers cannot
mutate a shared cached value. Feature views are deliberately not cached: slicing
the immutable input is already cheaper than copying a cached result.
"""
from collections import OrderedDict
from functools import wraps
import time
import weakref

import torch


class ExpressionEvaluationCache:
    """Context manager that restores upstream methods on exit; CPU, no-grad only."""
    def __init__(self, max_bytes=256 * 1024 * 1024, minimum_compute_seconds=0.0005):
        self.max_bytes = int(max_bytes)
        self.minimum_compute_seconds = float(minimum_compute_seconds)
        self.entries = OrderedDict()
        self.data_refs = {}
        self.originals = []
        self.bytes = 0
        self.counters = dict(hits=0, misses=0, inserts=0, evictions=0,
                             oversized=0, cheap_skips=0, bypasses=0)

    def stats(self):
        return dict(**self.counters, entries=len(self.entries), bytes=self.bytes,
                    max_bytes=self.max_bytes)

    def clear(self):
        self.entries.clear()
        self.data_refs.clear()
        self.bytes = 0

    def _key(self, expression, data, period):
        identity = id(data)
        existing = self.data_refs.get(identity)
        if existing is None or existing() is not data:
            # Python can recycle an object's id after collection. Never let an
            # older dataset's entries survive an identity reuse.
            if existing is not None:
                for key in [key for key in self.entries if key[0] == identity]:
                    previous = self.entries.pop(key)
                    self.bytes -= previous.numel() * previous.element_size()
            self.data_refs[identity] = weakref.ref(data)
        tensor = data.data
        return (identity, id(tensor), tensor._version, tuple(tensor.shape),
                type(expression), str(expression), period.start, period.stop, period.step)

    def _wrap(self, method):
        @wraps(method)
        def evaluate(expression, data, period=slice(0, 1)):
            if data.data.device.type != 'cpu' or data.data.requires_grad:
                self.counters['bypasses'] += 1
                return method(expression, data, period)
            key = self._key(expression, data, period)
            if key in self.entries:
                self.counters['hits'] += 1
                cached = self.entries.pop(key)
                self.entries[key] = cached
                return cached.clone()
            self.counters['misses'] += 1
            started = time.perf_counter()
            result = method(expression, data, period)
            seconds = time.perf_counter() - started
            # Never memoize autograd graphs or functions that are faster than copying.
            if result.requires_grad or seconds < self.minimum_compute_seconds:
                self.counters['cheap_skips'] += 1
                return result
            size = result.numel() * result.element_size()
            if size > self.max_bytes:
                self.counters['oversized'] += 1
                return result
            owned = result.detach().clone()
            while self.entries and self.bytes + size > self.max_bytes:
                _, previous = self.entries.popitem(last=False)
                self.bytes -= previous.numel() * previous.element_size()
                self.counters['evictions'] += 1
            self.entries[key] = owned
            self.bytes += size
            self.counters['inserts'] += 1
            return result
        return evaluate

    def __enter__(self):
        import alphagen.data.expression as expression_module
        for name, cls in vars(expression_module).items():
            if not isinstance(cls, type) or not issubclass(cls, expression_module.Expression):
                continue
            if cls in (expression_module.Expression, expression_module.Feature, expression_module.DeltaTime):
                continue
            if 'evaluate' in cls.__dict__:
                original = cls.__dict__['evaluate']
                self.originals.append((cls, original))
                cls.evaluate = self._wrap(original)
        return self

    def __exit__(self, *exception):
        for cls, original in reversed(self.originals):
            cls.evaluate = original
        self.originals.clear()

