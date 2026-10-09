# Exp 7 preregistration: a launch-aware model of batch-1 decode on a T4

*Written before any Exp 7 measurement. The commit date of this file is the evidence of that.*

## Question

For which model sizes is single-stream (batch of one) decoding on a Tesla T4 limited by the CPU launching GPU operations, and for which by the GPU itself? Exp 6 found that for Qwen2.5-1.5B the eager step is CPU-bound (the GPU is busy for 71% of it). Related work exists on the effect (a 2026 batch-1 study on larger GPUs and a 2025 CPU/GPU-boundness metric on CPU-GPU coupled systems), so this is a test of a simple predictive model on small models and an older GPU, not a claim of a new phenomenon.

## Model

For one decode step with a static cache and a batch of one:

```
T_gpu   = weight_bytes / bandwidth + layers * c_layer
T_cpu   = layers * ops_per_layer * t_launch
eager   = max(T_gpu, T_cpu)
graph   = T_gpu
```

The model is fitted on Qwen2.5-1.5B only: `c_layer` from its CUDA-graph step time and its weight-streaming time, `ops_per_layer` from the profiler's device operations per step, `t_launch` from its eager step time divided by that operation count. The bandwidth is the one measured on the reference run. Nothing from the other models' step times is used. `src/exp7_fit.py` defines the computation exactly.

## Predictions (from the Exp 6 numbers, before measuring)

| Model | Eager step | CUDA-graph step | Speed-up from a graph | Regime |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | about 21.5 ms | about 11.4 ms | about 1.9x | CPU-bound |
| Qwen2.5-3B | about 36 ms | about 36 ms | about 1.0x | GPU-bound, close to balance |

These come from rough parameter counts (0.49B and 3.09B) and the Exp 6 reference values (floor 12.7 ms, 304 microseconds of extra GPU time per layer, 50 operations per layer, 17.9 microseconds per operation). The final predictions are recomputed by the fit script from the new run of the reference model.

## Acceptance criteria

The model is accepted only if, for both the 0.5B and the 3B model:
1. the error of the predicted eager step and of the predicted graph step is at most 20%;
2. the predicted regime (CPU-bound or GPU-bound) equals the measured one, where a step counts as CPU-bound when the time to enqueue it is at least 95% of the step time;
3. the mean absolute error of the model is lower than that of both baselines: scaling the reference step time by weight bytes only, and by layer count only.

Otherwise the result is reported as a negative result, with the same prominence.

## Not claimed

No claim of a new phenomenon, one GPU type, one model family, fp16, batch of one, prompt of 512 tokens, 127 decode steps. A second GPU (for example a P100) is optional and would be reported separately.
