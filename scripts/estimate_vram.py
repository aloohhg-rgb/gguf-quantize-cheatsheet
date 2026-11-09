#!/usr/bin/env python3
"""Rough VRAM estimator for GGUF-style quants.

Uses approximate bytes/param and a simplified KV formula.
This will be wrong in the details — that's fine. It's a floor check.
"""
from __future__ import annotations

import argparse
import sys

# Approximate average bytes per parameter for common GGUF quants.
# K-quants vary by tensor; these are planning numbers, not measurements.
BYTES_PER_PARAM = {
    "F16": 2.0,
    "Q8_0": 1.05,
    "Q6_K": 0.78,
    "Q5_K_M": 0.68,
    "Q5_K_S": 0.65,
    "Q4_K_M": 0.58,
    "Q4_K_S": 0.53,
    "Q3_K_M": 0.46,
    "Q3_K_S": 0.42,
    "Q2_K": 0.36,
    "IQ4_XS": 0.50,  # ballpark — IQ family varies
    "IQ3_XXS": 0.35,
}


def estimate(
    params_b: float,
    quant: str,
    ctx: int,
    n_layers: int,
    hidden: int,
    overhead_gb: float,
    moe: bool,
    active_b: float | None,
) -> dict:
    q = quant.upper()
    if q not in BYTES_PER_PARAM:
        raise SystemExit(
            f"unknown quant {quant!r}. known: {', '.join(BYTES_PER_PARAM)}"
        )
    bpp = BYTES_PER_PARAM[q]
    weights_gb = params_b * bpp  # params_b is billions; bpp * 1e9 / 1e9 = bpp * params_b

    # Lazy KV: ctx * layers * hidden * 2(K+V) * 2(fp16) bytes
    kv_gb = (ctx * n_layers * hidden * 2 * 2) / 1e9

    total = weights_gb + kv_gb + overhead_gb
    out = {
        "quant": q,
        "bytes_per_param_approx": bpp,
        "weights_gb": weights_gb,
        "kv_gb_fp16_approx": kv_gb,
        "overhead_gb": overhead_gb,
        "total_gb_floor": total,
        "notes": [],
    }
    if moe:
        out["notes"].append(
            "MoE: weights use total params; KV formula still uses the layer/hidden you passed "
            "(usually closer to dense-of-active — verify arch)."
        )
        if active_b is not None:
            out["notes"].append(
                f"active_b={active_b} is recorded for your notes; this script does not "
                "scale FLOPs, only memory floors."
            )
    out["notes"].append(
        "Floor only. Fragmentation, CUDA/HIP context, and framework buffers add more."
    )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-b", type=float, required=True, help="total params in billions")
    p.add_argument("--quant", type=str, default="Q4_K_M")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=40, help="transformer layers (guess ok)")
    p.add_argument("--hidden", type=int, default=5120, help="hidden size (guess ok)")
    p.add_argument("--overhead-gb", type=float, default=2.0)
    p.add_argument("--moe", action="store_true")
    p.add_argument("--active-b", type=float, default=None, help="active params B (MoE note only)")
    args = p.parse_args(argv)

    est = estimate(
        params_b=args.params_b,
        quant=args.quant,
        ctx=args.ctx,
        n_layers=args.n_layers,
        hidden=args.hidden,
        overhead_gb=args.overhead_gb,
        moe=args.moe,
        active_b=args.active_b,
    )

    print(f"quant:            {est['quant']} (~{est['bytes_per_param_approx']} B/param)")
    print(f"weights:          {est['weights_gb']:.2f} GB")
    print(f"KV (fp16 approx): {est['kv_gb_fp16_approx']:.2f} GB  "
          f"[ctx={args.ctx}, layers={args.n_layers}, hidden={args.hidden}]")
    print(f"overhead pad:     {est['overhead_gb']:.2f} GB")
    print(f"TOTAL floor:      {est['total_gb_floor']:.2f} GB")
    for n in est["notes"]:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
