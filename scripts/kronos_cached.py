"""Exact causal-attention cache for released Kronos inference (eval, no rollover).

Weights, tokenizer, sampling, and the released dependency-aware head are unchanged.
The identical history is evaluated once before expanding Monte Carlo trajectories.
"""
import numpy as np
import torch
import torch.nn.functional as F


def offload_transformers(model, device='cuda:0', resident_layers=0):
    """Keep original CPU weights and transfer one block at a time, without quantizing."""
    for name, child in model.named_children():
        if name != 'transformer':
            child.to(device)
    for i, layer in enumerate(model.transformer):
        if i < resident_layers:
            layer.to(device)
            continue
        layer._cpu_parameters = [(p, p.data) for p in layer.parameters()]
        layer._cpu_buffers = [(module, name, value) for module in layer.modules()
                              for name, value in module._buffers.items() if value is not None]

        def load(block, *unused):
            for parameter, cpu in block._cpu_parameters:
                parameter.data = cpu.to(device)
            for module, name, cpu in block._cpu_buffers:
                module._buffers[name] = cpu.to(device)

        def unload(block, *unused):
            for parameter, cpu in block._cpu_parameters:
                parameter.data = cpu
            for module, name, cpu in block._cpu_buffers:
                module._buffers[name] = cpu

        layer._load_for_cache = load
        layer._unload_for_cache = unload
        layer.register_forward_pre_hook(load)
        layer.register_forward_hook(unload)
    return model


def last_s2_logits(model, context, s1):
    # The released head projects every context position and then discards all
    # but the last. The cross-attention query has length one, so its output is
    # shared across those residual positions; only the final residual is needed.
    dep = model.dep_layer
    sibling = model.embedding.emb_s1(s1)
    attended = dep.cross_attn(query=sibling,key=context,value=context)
    return model.head.cond_forward(dep.norm(context[:,-1:]+attended))[:,0]


class CachedBackbone:
    def __init__(self, model):
        if model.training:
            raise ValueError("Caching requires eval mode")
        self.model = model
        self.cache = []
        self.length = 0
        self.path_count = None

    def step(self, s1, s2, stamp):
        m = self.model
        x = m.embedding([s1, s2]) + m.time_emb(stamp)
        offset = self.length
        if offset and x.shape[1] != 1:
            raise ValueError("Incremental calls must contain one token")
        new_cache = []
        for i, layer in enumerate(m.transformer):
            if hasattr(layer, '_load_for_cache'):
                layer._load_for_cache(layer)
            h = layer.norm1(x)
            a = layer.self_attn
            b, n, _ = h.shape
            project = lambda p: p(h).view(b, n, a.n_heads, a.head_dim).transpose(1, 2)
            q, k, v = project(a.q_proj), project(a.k_proj), project(a.v_proj)
            positions = torch.arange(offset, offset + n, device=x.device).type_as(a.rotary.inv_freq)
            freqs = torch.einsum('i,j->ij', positions, a.rotary.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos()[None, None], emb.sin()[None, None]
            q = q * cos + a.rotary._rotate_half(q) * sin
            k = k * cos + a.rotary._rotate_half(k) * sin
            if offset and self.path_count:
                # Monte Carlo paths share the same historical K/V. Keep that
                # prefix once, rather than storing ten identical copies.
                prefix_k, prefix_v = self.prefix_cache[i]
                old_k, old_v = self.cache[i]
                if old_k is not None:
                    k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
                new_cache.append((k,v))
                paths = self.path_count
                q5 = q.reshape(b//paths,paths,a.n_heads,1,a.head_dim)
                k5 = k.reshape(b//paths,paths,a.n_heads,-1,a.head_dim)
                v5 = v.reshape(b//paths,paths,a.n_heads,-1,a.head_dim)
                scores = torch.cat((q5 @ prefix_k[:,None].transpose(-2,-1),
                                    q5 @ k5.transpose(-2,-1)),dim=-1) / (a.head_dim ** 0.5)
                weights = torch.softmax(scores,dim=-1)
                prefix_len = prefix_k.shape[2]
                h = (weights[...,:prefix_len] @ prefix_v[:,None] +
                     weights[...,prefix_len:] @ v5).reshape(b,a.n_heads,1,a.head_dim)
            elif offset:
                old_k, old_v = self.cache[i]
                k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
                new_cache.append((k,v))
                h = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
            else:
                new_cache.append((k,v))
                h = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
            h = h.transpose(1, 2).contiguous().view(b, n, a.d_model)
            x = x + a.out_proj(h)
            x = x + layer.ffn(layer.norm2(x))
            if hasattr(layer, '_unload_for_cache'):
                layer._unload_for_cache(layer)
        self.cache = new_cache
        self.length += s1.shape[1]
        x = m.norm(x)
        return m.head(x[:, -1:]), x

    def expand_paths(self, count):
        self.path_count = count
        self.prefix_cache = self.cache
        self.cache = [(None,None) for _ in self.prefix_cache]


@torch.inference_mode()
def cached_inference(tokenizer, model, x, x_stamp, y_stamp, pred_len=10,
                     sample_count=10, T=0.6, top_p=0.9):
    from model.kronos import sample_from_logits
    if x.shape[1] + pred_len > 512:
        raise ValueError("This cache does not implement context rollover")
    b = x.shape[0]
    tokens = tokenizer.encode(x.clamp(-5, 5), half=True)
    backbone = CachedBackbone(model)
    logits, context = backbone.step(tokens[0], tokens[1], x_stamp)
    backbone.expand_paths(sample_count)
    context = context.repeat_interleave(sample_count, 0)
    logits = logits.repeat_interleave(sample_count, 0)
    y_stamp = y_stamp.repeat_interleave(sample_count, 0)
    future_s1, future_s2 = [], []
    for i in range(pred_len):
        if i:
            logits, latest = backbone.step(s1, s2, y_stamp[:, i-1:i])
            context = torch.cat((context, latest), dim=1)
        s1 = sample_from_logits(logits[:, -1], temperature=T, top_k=0, top_p=top_p)
        logits2 = last_s2_logits(model,context,s1)
        s2 = sample_from_logits(logits2, temperature=T, top_k=0, top_p=top_p)
        future_s1.append(s1)
        future_s2.append(s2)
    full = [torch.cat((t.repeat_interleave(sample_count, 0), torch.cat(f, dim=1)), dim=1)
            for t, f in zip(tokens, (future_s1, future_s2))]
    z = tokenizer.decode(full, half=True)
    # Match the released NumPy float32 mean, including its reduction order.
    return z.cpu().numpy().reshape(b, sample_count, z.shape[1], z.shape[2]).mean(axis=1)[:, -pred_len:]


@torch.inference_mode()
def verify_cache(model, x_stamp):
    """Compare prefill and all ten incremental logits under forced token histories."""
    torch.manual_seed(7381)
    device = next(model.parameters()).device
    s1 = torch.randint(0, model.s1_vocab_size, (2, 100), device=device)
    s2 = torch.randint(0, 2 ** model.s2_bits, (2, 100), device=device)
    stamp = x_stamp[:1].expand(2, -1, -1).to(device)
    cache = CachedBackbone(model)
    context = None
    worst = dict(s1=0.0, context=0.0, s2=0.0)
    for n in range(90, 101):
        start = 0 if n == 90 else n-1
        got, h = cache.step(s1[:, start:n], s2[:, start:n], stamp[:, start:n])
        context = h if context is None else torch.cat((context, h), 1)
        ref, ref_context = model.decode_s1(s1[:, :n], s2[:, :n], stamp[:, :n])
        pairs = {'s1': (got, ref[:, -1:]), 'context': (context, ref_context),
                 's2': (model.decode_s2(context, s1[:, n-1:n]),
                        model.decode_s2(ref_context, s1[:, n-1:n]))}
        for name, (a, b) in pairs.items():
            worst[name] = max(worst[name], (a-b).abs().max().item())
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(last_s2_logits(model,context,s1[:,n-1:n]),
                                   pairs['s2'][1][:,-1],atol=1e-4,rtol=1e-4)
    return dict(status='passed', max_absolute_errors=worst, atol=1e-4, rtol=1e-4,
                lengths=list(range(90, 101)), batch=2, dtype=str(next(model.parameters()).dtype))
