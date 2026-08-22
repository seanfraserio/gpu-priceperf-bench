# The 27B does not boot on a 45GB L40S

Six of the seven matrix cells have three complete runs. The seventh —
Qwen3.8-27B on an L40S — has none, and not because the tier was unavailable.
The model does not start.

## What happens

vLLM 0.27.1, fp8, tensor-parallel 1, on a host reporting 44.40 GiB usable:

```
Model loading took 28.06 GiB memory and 14.6 seconds
...
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.53 GiB.
GPU 0 has a total capacity of 44.40 GiB of which 59.31 MiB is free.
```

The failure is in `profile_cudagraph_memory`, before any KV cache is
allocated. Weights account for 28.06 GiB; profiling the CUDA graphs for the
sweep's capture set consumes most of the remainder, and the card runs out.

## What it is not

The obvious suspect is the 32k context window, so that was tested directly: a
second run at `MAX_MODEL_LEN=8192` — still 6x more than the benchmark's own
1024-in/256-out requests ever use — failed **byte for byte identically**. Same
1.53 GiB allocation, same 59.31 MiB free. Context length is not the binding
constraint; the batch and CUDA-graph footprint is.

## What would change it

The graph capture set is sized by `max_num_seqs`, which this harness pins to
the top sweep level (512). A smaller ceiling would shrink the graphs and might
fit. That has not been run, deliberately: it would measure the L40S on a
different sweep from every other tier, and the comparison the report exists to
make is between tiers.

Anyone wanting that number should run the tier with a reduced level list and
label it as its own configuration rather than folding it into this matrix.

## The honest summary line

Under this harness's uniform configuration, the cheapest 48GB-class card on
Vast cannot serve this 27B at all. That is a real deployment constraint, not a
gap in the data.
