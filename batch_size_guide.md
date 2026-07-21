# Batch Size Tuning Guide for VAE Training

## What batch size controls

Batch size is the number of data rows processed together in one forward+backward pass before updating model weights. It affects three things: **memory usage**, **training speed**, and **gradient quality**.

## Memory: the hard constraint

Each batch must fit in memory (RAM for CPU, VRAM for GPU). The memory cost per batch is:

```
batch_memory ≈ batch_size × input_dims × bytes_per_element × multiplier
```

- `input_dims`: for our VAE, ~95 (80 embedding + 15 numeric)
- `bytes_per_element`: 4 (float32) or 2 (float16 with AMP)
- `multiplier`: ~4–6× for activations, gradients, and optimizer state (AdamW stores 2 extra copies of each parameter)

For our model (95 → 256 → 128 → 16 latent):

| Batch size | Tensor memory | Peak memory (with grads) | Fits on... |
|-----------|--------------|-------------------------|------------|
| 4,096 | ~1.5 MB | ~10 MB | anything |
| 8,192 | ~3 MB | ~20 MB | anything |
| 32,768 | ~12 MB | ~80 MB | anything |
| 131,072 | ~50 MB | ~300 MB | anything with 1GB+ |
| 1,000,000 | ~380 MB | ~2 GB | 6GB+ GPU, any CPU |

Our model is tiny. The batch size limit for this architecture is effectively unbounded on CPU (96GB) and ~500K+ on a 6GB GPU. For larger models (ResNets, transformers), batch size becomes a real constraint.

**What happens if you exceed memory:**
- **GPU**: immediate crash with `torch.cuda.OutOfMemoryError`. No checkpoint saved.
- **CPU**: OOM killer terminates the process, or the system swaps to disk and the job becomes ~100× slower before SLURM kills it for exceeding time.

## Speed: why larger is faster (up to a point)

The training loop has per-batch overhead:
1. Python loop iteration + DataLoader yielding a batch
2. Data transfer to GPU (if applicable)
3. Optimizer step (weight update)
4. CUDA kernel launch overhead (GPU)

These costs are **per batch, not per row**. Doubling batch size halves the number of batches per epoch, cutting overhead roughly in half.

For our 89M-row training set:

| Batch size | Batches/epoch | Estimated epoch time (CPU) |
|-----------|--------------|---------------------------|
| 4,096 | 21,768 | ~32 min |
| 8,192 | 10,884 | ~20 min |
| 32,768 | 2,721 | ~10–12 min |
| 131,072 | 681 | ~8–10 min |

Diminishing returns set in around 32K–64K because the actual matrix math (forward/backward) starts dominating over Python overhead.

## Gradient quality: the training tradeoff

Smaller batches produce **noisier gradient estimates**, which acts as implicit regularization. Larger batches produce **smoother gradients** that point more accurately toward the loss minimum.

### The critical batch size

Empirically, there is a "critical batch size" below which doubling batch size gives near-linear speedup with no quality loss, and above which you start needing compensating adjustments.

McCandlish et al. (2018) formalize this as:

```
B_crit = B_noise / (1 + B_noise / B_simple)
```

where `B_noise` measures gradient noise. Below `B_crit`, larger batches are free speed. Above it, you're wasting compute.

For tabular VAEs on large datasets (89M rows), `B_crit` is typically very high (50K–200K) because:
- The data is heterogeneous (many categorical features with diverse values)
- The model is small (few parameters relative to data)
- Each batch already sees a representative sample of the data distribution

### The linear scaling rule

When increasing batch size beyond `B_crit`, scale the learning rate proportionally:

```
new_lr = old_lr × (new_batch_size / old_batch_size)
```

This was established by Goyal et al. (2017) for SGD and applies approximately to Adam/AdamW. Combine with a **warmup** period (5–10% of training) to avoid instability at the start.

For our setup: going from 4096 → 32768 (8× increase) is still well below the critical batch size for 89M rows, so the default learning rate (1e-3) should work without adjustment.

### When large batches hurt

- **Small datasets** (< 100K rows): large batches can cause overfitting because each batch is too representative — the model doesn't benefit from stochastic noise.
- **Very large batches relative to dataset size**: if `batch_size > dataset_size / 10`, you're effectively doing full-batch gradient descent, losing regularization.
- **GANs and RL**: these have inherently unstable optimization where gradient noise is beneficial.

None of these apply to our case (89M rows, batch 32K = 0.04% of dataset per batch).

## Practical recommendations for this pipeline

| Dataset size | Recommended batch size | Notes |
|-------------|----------------------|-------|
| < 100K rows | 1,024–2,048 | noise is beneficial for regularization |
| 100K – 1M | 4,096 | safe default |
| 1M – 10M | 4,096–8,192 | can go higher, diminishing returns |
| 10M – 100M | 8,192–32,768 | main benefit is speed, not quality |
| 100M+ | 32,768–65,536 | Python overhead dominates below this |

Current pipeline setting: **32,768** for ~89M filtered training rows.

## How to diagnose batch size issues

**Too large (rare for tabular data):**
- Val loss plateaus much higher than with smaller batch
- Training loss drops very fast but val loss diverges → overfitting
- Fix: reduce batch size or increase learning rate warmup

**Too small:**
- Training is slow (many batches per epoch, high Python overhead)
- Gradients are noisy → loss curve is jagged
- Fix: increase batch size, no other adjustments needed below `B_crit`

## References

1. **Goyal et al. (2017)** — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." Establishes the linear scaling rule for learning rate with batch size. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

2. **McCandlish et al. (2018)** — "An Empirical Model of Large-Batch Training." Defines the critical batch size and gradient noise scale. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

3. **Smith et al. (2018)** — "Don't Decay the Learning Rate, Increase the Batch Size." Shows batch size increase is mathematically equivalent to learning rate decay. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489)

4. **Hoffer et al. (2017)** — "Train longer, generalize better: closing the generalization gap in large batch training of neural networks." Addresses the generalization gap for large batches. [arXiv:1705.08741](https://arxiv.org/abs/1705.08741)

5. **Masters & Luschi (2018)** — "Revisiting Small Batch Training for Deep Neural Networks." Empirically shows small batches (2–32) often generalize better than large ones, but this effect diminishes with very large datasets. [arXiv:1804.07612](https://arxiv.org/abs/1804.07612)

6. **Keskar et al. (2017)** — "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." Shows large batches tend to converge to sharp minima with worse generalization — but this is primarily a concern for image classification, less so for tabular autoencoders. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
