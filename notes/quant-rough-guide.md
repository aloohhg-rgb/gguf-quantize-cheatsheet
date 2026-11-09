# quant letter soup (rough)

GGUF K-quants mix higher/lower precision across tensor types. Names are hints, not contracts.

| name | vibe |
|------|------|
| Q8_0 | near-fp16 quality for many tasks, fat |
| Q6_K | usually "I want quality but not Q8" |
| Q5_K_M / Q5_K_S | S = smaller/simpler, M = mixed (often preferred) |
| Q4_K_M | my default starting point for mid-size models |
| Q4_K_S | leaner Q4, sometimes noticeably worse |
| Q3_K_M | when VRAM is angry |
| Q2_K | desperation / experiments |

**I_Q** / **IQ** quants (importance-matrix) are a different family — often better quality at similar size, but conversion is another rabbit hole. Not covered deeply here yet.

Pick by: (1) does it fit with the context you actually use, (2) then climb quality if you have headroom. Don't chase leaderboard deltas before the thing loads.
