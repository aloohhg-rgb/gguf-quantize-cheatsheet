# gguf-quantize-cheatsheet

Personal cheatsheet / scratchpad for picking GGUF quants without lying to myself about VRAM.

I kept mental-math-ing "Q4_K_M on a 27B should fit on 24GB…" and then OOMing because I forgot KV cache / context / overhead. This folder is the formulas + a tiny estimator script. **No fake bench numbers.** If you want quality rankings, go read r/LocalLLaMA threads — they rot faster than I can copy them.

## files

| path | what |
|------|------|
| `notes/quant-rough-guide.md` | what the letter soup roughly means |
| `notes/vram-math.md` | honest-ish memory estimate formulas |
| `scripts/estimate_vram.py` | CLI that applies those formulas |
| `TODO.md` | stuff I still want to add |

## estimate (no model download needed)

```bash
# dense 27B, Q4_K_M, 8k context, llama-ish
python3 scripts/estimate_vram.py --params-b 27 --quant Q4_K_M --ctx 8192

# MoE: pass active params for compute vibes, total params for weight RAM
python3 scripts/estimate_vram.py --params-b 26 --active-b 4 --quant Q4_K_M --ctx 4096 --moe
```

Output is a **rough floor**, not a guarantee. Drivers, frameworks, and fragmentation all eat extra.

## quick gut feel (weights only, ignore KV)

Rough bytes/param (order of magnitude, K-quants vary by tensor):

| quant | ~bytes/param |
|-------|--------------|
| Q8_0  | ~1.0 |
| Q6_K  | ~0.75 |
| Q5_K_M| ~0.65 |
| Q4_K_M| ~0.55 |
| Q3_K_M| ~0.45 |
| Q2_K  | ~0.35 |

`weights_GB ≈ params_B * bytes_per_param`

Then add KV cache (see `notes/vram-math.md`). Then add ~1–3 GB for runtime overhead depending on stack.

## license

MIT. Not affiliated with llama.cpp / ggml folks — just a user with a bad memory.
