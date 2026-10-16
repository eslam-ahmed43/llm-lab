# Exp 7b preregistration: does the model predict the switch to GPU-bound?

*Written before any Exp 7b measurement. The commit date of this file is the evidence of that.*

## Why a second experiment

Exp 7 measured Qwen2.5 0.5B, 1.5B and 3B in fp16 on a Tesla T4 and tested the model fitted on the 1.5B (see `PREREGISTRATION_EXP7.md`). By the preregistered criteria the model was accepted: with the predictions recomputed from the new reference run, the errors of the eager and graph steps were between 0.0% and 9.4%, and the mean absolute error was 2.8% against 27.0% (layer count only) and 41.5% (weight bytes only).

That result has a limit that matters. All three sizes were CPU-bound, so the `max()` in the model, the part that predicts a switch to GPU-bound, was never exercised. In the CPU-bound regime the eager prediction of the model equals the layer-count baseline by construction, and only the graph step separates the models. Exp 7 also showed that the eager step depends on the host: the same 1.5B model took 25.1 ms in one session and 30.2 ms in the next, while its graph step moved from 21.2 to 20.4 ms.

## Test

Move the GPU time up without changing the number of layers or operations: run the same models with fp32 weights, which doubles the bytes read per token. The model predicts that the larger fp32 models become GPU-bound. To remove the host effect, the fp16 1.5B reference is measured again in the same session and the model is refitted on it. `src/exp7_fit.py` computes the predictions.

## Predictions (computed from the fitted values of Exp 7, before measuring)

| Run | Weights | Eager step | CUDA-graph step | Speed-up from a graph | Regime |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B, fp32 | 1.98 GB | about 25.9 ms | about 14.7 ms | about 1.8x | CPU-bound |
| Qwen2.5-1.5B, fp32 | 6.17 GB | about 33.2 ms | about 33.2 ms | about 1.0x | GPU-bound (close to the boundary: CPU time about 30.2 ms) |
| Qwen2.5-3B, fp32 | 12.34 GB | about 61.0 ms | about 61.0 ms | about 1.0x | GPU-bound (CPU time about 38.8 ms) |

The final predictions are recomputed by the fit script from the same-session reference. The 1.5B fp32 case is the hardest: the two sides differ by about 10%, so a slower host could flip it.

## Acceptance criteria (as in `PREREGISTRATION_EXP7.md`, evaluated by `exp7_fit.py` on all runs except the reference)

1. every predicted eager and graph step is within 20% of the measurement;
2. the predicted regime equals the measured regime (a step counts as CPU-bound when the time to enqueue it is at least 95% of the step time);
3. the mean absolute error of the model is lower than that of the weight-bytes-only baseline and of the layer-count-only baseline.

Additional statement made in advance: the prediction that the 3B fp32 run is GPU-bound is the main test of the `max()` structure. If it is measured as CPU-bound, the two-term structure is not supported at this scale, whatever the other criteria say.

## Assumption that may fail

The per-layer GPU term `c_layer` is fitted on fp16 and assumed to carry over to fp32. If it does not, the errors in the GPU-bound regime will be larger. That would be reported as a finding about the model, not hidden.

## Not claimed

One GPU type, one model family, batch of one, 512-token prompt, 127 decode steps, weights stored in fp32 as a way to change the bytes read (not a recommended deployment). Results on other hosts may differ, since the CPU-side term depends on the host.
