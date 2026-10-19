# Exp 7: A launch-aware model of batch-1 decode on a T4

*Qwen2.5 0.5B, 1.5B and 3B on a Kaggle Tesla T4, fp16 and fp32. Exp 7 and Exp 7b, both preregistered. Last updated 2026-10-17.*

## Abstract

For single-stream decoding, the time of one step is either set by the GPU (reading the weights) or by the CPU (launching about a thousand GPU operations). We tested a two-term model, `eager step = max(GPU time, CPU time)` and `CUDA-graph step = GPU time`, fitted on one reference model (Qwen2.5-1.5B in fp16) and used to predict the other runs with predictions and acceptance criteria written down before measuring. Across seven predicted runs (0.5B, 1.5B and 3B in fp16 and fp32) the model's mean absolute error was about 3%, against 27% to 41% for two single-factor baselines. In Exp 7 (three fp16 sizes) all preregistered criteria were met, but every run was CPU-bound, so the switch to GPU-bound was not exercised. Exp 7b raised the GPU time by storing the weights in fp32. The predicted step times were within 1% for the two largest runs, but **the preregistered criterion on the regime failed, so by our own rules the model is not accepted.** The failure comes from how we defined the regime: when the GPU is the bottleneck the CPU blocks on a full launch queue, so enqueue time equals step time and the rule labels a GPU-bound step as CPU-bound. A post-hoc analysis with indicators that do not have this flaw gives the predicted regimes, but it is exploratory and needs a new preregistered test. Setting aside the verdict, the measurements also show that the eager step varies by about 20% between sessions on different hosts while the graph step varies by about 4%.

## 1. Related work

A 2026 study of batch-1 decoding (J. Chen, arXiv 2605.30571, "Memory-Bound but Not Bandwidth-Limited") reports on larger GPUs that decode latency is driven by kernel-launch overhead more than by raw memory bandwidth, with about 1.26x from CUDA graphs on an H100 and a launch overhead that stays mostly hidden on slower GPUs for 7 to 8B models. A 2025 study of CPU-GPU coupled systems proposes a metric to find the transition between CPU-bound and GPU-bound inference (reference details to be verified before publication). The effect of CUDA graphs on LLM decoding is also described in an engineering blog post by Fireworks AI ("How CUDA Graphs Enable Fast Python Code for Deep Learning"). We do not claim a new phenomenon. What we add is a simple predictive model tested on small models and an older GPU, with the predictions registered in advance (`PREREGISTRATION_EXP7.md`, `PREREGISTRATION_EXP7B.md`).

## 2. Method

**Decode step.** The from-scratch decoder of Exp 5 with the static-cache step of Exp 6, batch of one, a 512-token prompt, 127 timed decode steps, one warm-up and five repetitions. We time the eager step with a preallocated cache and the same step captured as a CUDA graph. We also record the CPU time needed to enqueue the step, the number of device operations per step and the GPU busy time (`torch.profiler`, 30 steps), the parameter count, the weight bytes and the memory bandwidth measured with a large copy.

**Model.**

```
T_gpu = weight_bytes / bandwidth + layers * c_layer
T_cpu = layers * ops_per_layer * t_launch
eager = max(T_gpu, T_cpu)        graph = T_gpu
```

The three constants are fitted on the reference run only: `c_layer` from its graph step and its weight-streaming time, `ops_per_layer` from its profiled operation count, `t_launch` from its eager step time divided by that count. The other runs are predicted without using their step times. Baselines scale the reference step time by weight bytes only, or by layer count only.

**Preregistered criteria**, evaluated on the non-reference runs: (1) every predicted eager and graph step within 20% of the measurement; (2) the predicted regime equals the measured regime, where a step counts as CPU-bound when the time to enqueue it is at least 95% of the step time; (3) the model's mean absolute error is lower than that of both baselines. In `PREREGISTRATION_EXP7B.md` we also stated that if the 3B fp32 run is measured as CPU-bound, the two-term structure is not supported at this scale.

**Environment.** Tesla T4, Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1. Exp 7 and Exp 7b were run in different Kaggle sessions, and in Exp 7b the reference was measured again in the same session as the test runs.

## 3. Results

### 3.1 Exp 7: three sizes in fp16 (fitted on the 1.5B)

| Run | Eager step measured / predicted (ms) | Error | Graph step measured / predicted (ms) | Error | Graph speed-up measured / predicted |
|---|---:|---:|---:|---:|---:|
| 0.5B | 25.43 / 25.88 | +1.8% | 9.72 / 10.64 | +9.4% | 2.61 / 2.43 |
| 1.5B (fitted) | 30.20 | | 20.43 | | 1.48 |
| 3B | 38.71 / 38.83 | +0.3% | 35.41 / 35.41 | 0.0% | 1.09 / 1.10 |

Mean absolute error of the four predicted steps: model 2.8%, layer-count baseline 27.0%, weight-bytes baseline 41.5%. All three criteria were met. All three sizes were CPU-bound. Fitted constants: 272 microseconds of extra GPU time per layer, 49.9 operations per layer, 21.6 microseconds per operation.

### 3.2 Exp 7b: fp32 weights (fitted on the 1.5B fp16 measured again in the same session)

| Run | Eager step measured / predicted (ms) | Error | Graph step measured / predicted (ms) | Error | Graph speed-up measured / predicted | Regime measured / predicted |
|---|---:|---:|---:|---:|---:|---|
| 0.5B fp32 | 20.38 / 23.23 | +14.0% | 14.24 / 14.11 | -0.9% | 1.43 / 1.65 | CPU-bound / CPU-bound |
| 1.5B fp16 (fitted) | 27.10 | | 19.70 | | 1.38 | CPU-bound |
| 1.5B fp32 | 32.12 / 32.44 | +1.0% | 32.59 / 32.44 | -0.4% | 0.99 / 1.00 | **CPU-bound** / GPU-bound |
| 3B fp32 | 59.88 / 59.90 | 0.0% | 59.84 / 59.90 | +0.1% | 1.00 / 1.00 | **CPU-bound** / GPU-bound |

Mean absolute error of the six predicted steps: model 2.7%, layer-count baseline 31.2%, weight-bytes baseline 38.1%. Fitted constants: 248 microseconds per layer, 49.9 operations per layer, 19.4 microseconds per operation.

**Preregistered verdict for Exp 7b: criterion 1 passed, criterion 2 failed, criterion 3 passed. The model is not accepted.** The pre-stated main test (the 3B fp32 regime) was measured as CPU-bound by the preregistered rule, so by that rule the two-term structure is not supported.

### 3.3 Why criterion 2 failed: a flaw in the regime definition (exploratory)

The regime was measured as "enqueue time is at least 95% of the step time". That is valid when the GPU keeps up and the CPU is the bottleneck. When the GPU is the bottleneck, the launch queue fills, the CPU thread blocks, and the time to enqueue approaches the step time anyway. The measurements show that the rule cannot tell the two cases apart:

| Run (fp32 unless noted) | Enqueue time / step time (eager) | GPU busy / step time (eager) | Speed-up from a graph |
|---|---:|---:|---:|
| 0.5B | 99.9% | 68% | 1.43 |
| 1.5B fp16 (reference) | 99.9% | 70% | 1.38 |
| 1.5B | 99.3% | 96% | 0.99 |
| 3B | 99.4% | 98% | 1.00 |

The enqueue ratio is above 99% in every run, including the two where the GPU is busy 96% to 98% of the step and a CUDA graph does not help at all (speed-up 0.99 to 1.00). By those two indicators, which are physical and do not depend on the blocking behaviour, the 1.5B and 3B fp32 runs are GPU-bound, as the model predicted, and the 0.5B fp32 run is CPU-bound, as predicted. **This is a post-hoc reading.** We chose the new indicators after seeing the data, so it does not turn the preregistered failure into a pass. A confirmation would need a new preregistration that defines the regime by the graph speed-up or by the GPU busy fraction, and new predictions.

Two smaller observations, also post-hoc. First, the number of operations per layer depends on the precision: about 42 in fp32 (1008, 1172 and 1500 operations for 24, 28 and 36 layers) against about 50 in fp16 (the fp16 path has extra cast kernels), while the model assumed 49.9. This explains most of the 14% eager error of the 0.5B fp32 run: with its own operation count (1008) the CPU term becomes 19.6 ms, about 4% below the measured 20.38 ms. Second, the per-layer GPU term fitted on fp16 carried over to fp32 without a visible loss (graph steps within 1% for the two GPU-bound runs).

### 3.4 Host dependence

The same Qwen2.5-1.5B fp16 model, with the same code, in three Kaggle sessions:

| Session | Eager step (ms) | Graph step (ms) | CPU time per operation (microseconds) |
|---|---:|---:|---:|
| Exp 6 | 25.08 | 21.22 | 17.9 |
| Exp 7 | 30.20 | 20.43 | 21.6 |
| Exp 7b | 27.10 | 19.70 | 19.4 |

The eager step varied by about 20% and the graph step by about 8% from the highest to the lowest value (about 4% around the mean). The CPU side of the model depends on the host, which is why the reference had to be measured again in the same session for Exp 7b. It also moved the preregistered predictions: the approximate table written before Exp 7 predicted a 0.5B eager step of 21.5 ms, and it was 25.4 ms.

## 4. Analysis

1. **The two-term model predicts step times well**, with a mean absolute error of about 3% over seven predicted runs, against 27% to 41% for either single-factor baseline. Most of its advantage over the layer-count baseline comes from the graph step and the GPU-bound runs, because in the CPU-bound regime its eager prediction equals the layer-count baseline by construction.
2. **The switch to GPU-bound happens where the model puts it.** At 1.5B in fp32 the model puts the GPU time (32.4 ms) about 8% above the CPU time (about 30 ms), and the measurement shows the eager and graph steps equal (32.1 and 32.6 ms). At 3B in fp32 they are equal as predicted (59.9 and 59.8 ms). In fp16 on this T4 with this host, even a 3B model is still CPU-bound (graph speed-up 1.09).
3. **Our preregistered regime test was badly specified.** The preregistered result stands: the model is not accepted. The lesson is that an operational definition has to be validated against a case where the answer is known before it is preregistered.
4. **Eager decoding time depends on the machine**, graph time much less. Anyone comparing eager decoding speeds across machines is also comparing CPUs.

## 5. Limitations

- One GPU type, one model family, one host family (Kaggle, with visible variation between sessions), batch of one, a 512-token prompt and 127 steps.
- fp32 weights are used as a way to raise the GPU time, not as a recommended deployment. The GPU-bound regime was reached only this way.
- The criterion on the regime was badly defined (section 3.3). The post-hoc reading is exploratory and has not been confirmed on new data.
- The operation count per layer depends on the precision, and the model took it from the fp16 reference. A model for another architecture or framework would need its own count.
- The profiler's GPU busy time is not exact for very short kernels (it exceeded the step time in some runs, by up to about 22% for the 0.5B graph step). The model only uses the operation count from the profiler; the busy fractions in section 3.3 should be read as approximate.
- The approximate predictions written in the preregistration before Exp 7 were off by up to 18% in absolute terms because the host changed between sessions. The formal procedure (predictions recomputed from the reference measured in the same session) is what the criteria use.

## 6. Reproducibility

Raw results are in `results/launch_*.json` (Exp 7) and `results/launch7b_*.json` (Exp 7b), with the fits in `results/launch_model_fit.json` and `results/launch7b_fit.json`. The preregistration files carry their commit dates (`PREREGISTRATION_EXP7.md` and `PREREGISTRATION_EXP7B.md`, committed before the corresponding measurements).

```bash
python src/exp7_launch_model.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/launch7b_1p5b_fp16.json
python src/exp7_launch_model.py --model Qwen/Qwen2.5-3B-Instruct --dtype fp32 --out results/launch7b_3b_fp32.json
python src/exp7_fit.py --results-dir results --prefix launch7b --reference 1.5B --reference-dtype fp16 \
    --out results/launch7b_fit.json --plot plots/launch7b_model.png
```

## 7. What comes next

- A confirmation with a new preregistration that defines the regime by the graph speed-up (at most 1.05 means GPU-bound) or by the GPU busy fraction (at least 90%), with fresh predictions, ideally in a new session and on another GPU.
- Replace the fixed operation count per layer by one that depends on the precision, and test whether the model transfers to another architecture.
- Build the model into a small tool that measures a few constants on the user's machine and says whether a CUDA graph is worth trying.
