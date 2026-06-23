# Training Report — 2026-04-02

## Summary

Ran contrastive fine-tuning of text-embedding-mxbai-embed-large-v1 with GradCache + hard negatives.
Fixed 3 bugs in the training pipeline, ran ablation experiments on hyperparameters,
and completed a 1000-step training run.

## Bug Fixes

### 1. NCCL Deadlock (critical)

`prefetched()` used a `ThreadPoolExecutor` for background data loading.
Background threads conflicted with NCCL all-reduce, causing random deadlocks in multi-GPU training.

**Fix**: Removed prefetch, iterate `train_loader` directly. No performance impact (~8.2s/step before and after).

### 2. Processor max_pixels Misconfigured

`--max-num-visual-tokens` default was 1024, and we added code to set
`processor.image_processor.max_pixels = max_num_visual_tokens * 784`.
However, the processor's grid alignment means it doesn't strictly enforce this limit:
an 875×1024 image produces 3000 tokens regardless of the setting.

The default of 1024 actually *degraded* image quality slightly (3000 vs 3456 tokens)
without achieving precise token control.

**Fix**: Changed default to 4096. At this value, 875×1024 images produce 3456 tokens
(same as processor default), so no quality loss.

### 3. Retrieval Eval NCCL Timeout

`--test-max-pairs` defaulted to 0 (unlimited). With large test sets (e.g., 4575 queries × 36430 docs),
retrieval eval runs only on rank 0 while other ranks wait at a barrier.
This exceeded the 60-minute NCCL timeout.

**Fix**: Changed `--test-max-pairs` default from 0 to 500.

## 1000-Step Training Run

**Config**: 2×H100, batch_size=16, lr=1e-5, cosine schedule, warmup=20, 5 hard negatives per sample.

**Data**: 2400 pairs from `train_hn.jsonl` (1833 with hard negatives).

**Checkpoints**: `output_1000steps/checkpoint-{100..1000}`

| Step | Eval Loss | Eval Acc |
|------|-----------|----------|
| 50   | 0.613     | 80.6%    |
| 100  | 0.502     | 82.5%    |
| 200  | 0.445     | 82.8%    |
| 300  | 0.428     | 83.4%    |
| 400  | 0.421     | 84.4%    |
| 500  | 0.418     | 83.8%    |
| 600  | 0.414     | 83.8%    |
| 700  | 0.411     | 84.4%    |
| 800  | 0.410     | 84.4%    |
| 900  | 0.406     | 84.1%    |
| 1000 | 0.407     | 83.8%    |

Eval loss converges around step 400–500. Eval acc saturates at ~84.4% (step 400/700/800).
Best checkpoint region: **step 400–700**.

wandb: https://wandb.ai/andylizf-university-of-california-berkeley/wiki-screenshot-training/runs/jswdrobj

**Note**: No retrieval eval (R@1/R@5) was run — we had no `test.jsonl` and used `--test-eval-steps 0`.
Eval acc is contrastive accuracy on the validation set, not retrieval recall.

## Ablation Experiments (200 steps)

All compared against baseline: batch_size=16, lr=1e-5, temp=0.07, 2×H100.

| Experiment | Change | Eval Loss | Eval Acc | Verdict |
|------------|--------|-----------|----------|---------|
| **baseline** | — | 0.489 | 82.5% | best |
| bigbatch | batch 16→32 | 0.670 | 80.0% | worse (lr too low for larger batch) |
| lowlr | lr 1e-5→3e-6 | 0.609 | 81.3% | slightly worse |
| hightemp | temp 0.07→0.15 | 1.091 | 78.1% | worst |

wandb links:
- bigbatch: https://wandb.ai/andylizf-university-of-california-berkeley/wiki-screenshot-training/runs/3938ypgz
- lowlr: https://wandb.ai/andylizf-university-of-california-berkeley/wiki-screenshot-training/runs/ym4epldm
- hightemp: https://wandb.ai/andylizf-university-of-california-berkeley/wiki-screenshot-training/runs/5lmvi76m

## Train Loss Oscillation

Train loss oscillates between 1.5–6.0 throughout training without a clear downward trend.
This is expected in contrastive learning with small batches — each step's loss depends heavily
on which hard negatives are sampled. Eval loss (averaged over 20 batches) is the stable metric
and decreases monotonically.

Increasing batch size did not reduce oscillation (and hurt accuracy, likely because lr wasn't scaled up).

## Open Questions

1. **No retrieval eval done** — eval acc measures contrastive accuracy, not actual retrieval performance (R@1 etc.). Need to run retrieval eval on checkpoints to confirm real-world improvement.
2. **max_num_visual_tokens** doesn't precisely control token count due to processor grid alignment. The parameter is approximate at best.
3. **Data scale** — we trained on 2400 pairs. Wang Yichuan is training on 76K+. Results may differ significantly at scale.
