# VRAM math (honest-ish)

## weights

```
weights_bytes ≈ n_params * bytes_per_param(quant)
```

`bytes_per_param` for K-quants is **approximate** because different tensors use different sub-types. Good enough for "will this even load".

## KV cache (very rough, llama-style)

For a transformer with `n_layers` layers, `n_kv_heads` KV heads, `head_dim`:

```
# per token, both K and V, all layers (fp16 KV):
kv_per_token ≈ 2 * n_layers * n_kv_heads * head_dim * 2_bytes

# for ctx tokens:
kv_total ≈ kv_per_token * ctx
```

If you don't know arch details, a lazy fallback some people use:

```
kv_GB ≈ ctx * n_layers * hidden_size * 2 * 2 / 1e9
#                             ^K+V   ^fp16
```

GQA/MQA shrinks this vs full MHA. MoE usually KV-scales with **active** depth similar to a dense model of the active size — but check the arch.

## overhead

Runtime (llama-server / vLLM / etc.), CUDA|HIP context, fragmentation: budget **extra GB**. I mentally add 1–3 GB for llama.cpp-ish stacks; more if the framework is heavier.

## MoE

- **Weight RAM** tracks **total** params (all experts resident) unless you offload experts.
- **Compute / KV** closer to **active** params.

So a 26B-A4B MoE Q4 can look small on FLOPs but still need ~total-param weight memory if everything sits on GPU.

## what this is not

- Not a benchmark.
- Not a quality ranking.
- Not a promise your exact model fits.
