## V8 Ablation Study (2026-04-11/12)

### Goal
Find the best training recipe for v8 QA score (400 questions, 7426 tiles).

### Baseline
text-embedding-mxbai-embed-large-v1 (untrained): **QA = 0.715-0.720**

### Current SOTA
**v8o = v8r = QA 0.780** at step 200/150
- v8o: warmup100 hardswitch, lr=7e-6, cosine 400 steps, batch=64, hn=2, **lora_r=32 + ViT LoRA**
- v8r: warmup50 hardswitch, lr=7e-6, cosine 350 steps, batch=64, hn=2, **lora_r=32 + ViT LoRA**
- Best checkpoint: `training/output_nvme/v8_o_warmup100_lr7e6_lora_vit/checkpoint-200`

### LoRA Architecture Comparison (key finding)

| Config | Trainable Params | Best QA | Experiments |
|--------|-----------------|---------|-------------|
| r=32 LLM attn only | 12.8M | 0.7625 | v8i |
| r=32 LLM attn + MLP | 34.9M | TBD | v8t (running) |
| r=64 LLM attn only | 25.7M | 0.770 | v8s |
| **r=32 LLM attn + ViT attn+MLP** | **25.4M** | **0.780** | **v8o, v8r** |
| r=64 LLM attn + ViT attn+MLP | ~50M | **TODO** | **next experiment** |

**Key insight**: ViT LoRA (+2% QA) > larger LLM rank (+0.75% QA) at similar param count (~25M).
Qwen3-VL ViT uses fused `qkv` naming — default LoRA misses it entirely. `--lora-vit` fixes this.

### TODO: Next Experiments
- [ ] **r=64 + ViT LoRA (~50M params)** — combine both best improvements
- [ ] r=64 + ViT LoRA + lora-mlp — maximum capacity
- [ ] r=64 + ViT LoRA + different lr (may need lower lr with more params)

### Round 1: Recipe Comparison (LLM-only, r=32)

All: 1GPU, batch=64, hn=2, lora_r=32, lr=7e-6 unless noted.

| Experiment | Warmup | Recipe | Steps | Best QA | Peak Step |
|------------|--------|--------|-------|---------|-----------|
| **v8i** | **50** | **hardswitch** | **350** | **0.7625** | **200** |
| v8a | 100 | hardswitch | 400 | 0.755 | 150 |
| v8f | 0 | no warmup | 250 | 0.755 | 50 |
| v8c | 100 | mix 0.2 | 400 | 0.7525 | 250 |
| v8d | 50 | mix 0.3 | 350 | 0.750 | 200 |
| v8b | 100 | curriculum | 400 | 0.740 | 250 |
| v8e | 100 | curriculum (lr=5e-6) | 400 | 0.7375 | 150 |
| v8g | 100 | curriculum | 500 | 0.750 | 300 |

### Round 2: Variations (LLM-only, r=32)

| Experiment | Change | Best QA |
|------------|--------|---------|
| v8h | warmup100, 500 steps | 0.7525 |
| v8j | warmup100, lr=1e-5 | 0.7525 |
| v8k | warmup100, 600 steps | 0.755 |
| v8l | no warmup, 400 steps | 0.7425 |
| v8m | warmup100, hn=5 | 0.755 |

### Round 3: LoRA Architecture

| Experiment | Config | Params | Best QA | Status |
|------------|--------|--------|---------|--------|
| **v8o** | **warmup100 + ViT LoRA r=32** | **25.4M** | **0.780** | **done** |
| **v8r** | **warmup50 + ViT LoRA r=32** | **25.4M** | **0.780** | **running** |
| **v8s** | warmup50 + r=64 LLM-only | 25.7M | 0.770 | running |
| v8t | warmup50 + lora-mlp r=32 | 34.9M | TBD | running |
| v8p | warmup50 + 500 steps | 12.8M | 0.7525 | running |
| v8q | warmup50 + lr=5e-6 | 12.8M | 0.7525 | running |
| v8u | warmup25 | 12.8M | 0.740 | running |

### Special: Text-Only Training

| v8n | text-only 200 steps | **QA = 0.7025 (WORSE than 0.715 baseline)** |

### Key Findings

1. **ViT LoRA is the biggest improvement**: +2% QA over LLM-only at same param count
2. **LoRA rank matters**: r=64 > r=32 (+0.75% QA)
3. **warmup50 hardswitch is optimal recipe** for scheduling
4. **Hardswitch > curriculum > mix interleaving**
5. **lr=7e-6 is optimal**, cosine 350-400 steps
6. **Text-only training hurts** image QA (embedding drift)
7. **Peak QA at step 150-200**, checkpoint there
8. **Single GPU batch=64 > 4GPU batch=256**

---

## Colin Machine Reproduction (2026-04-12)

Reproduced on 8x H100 80GB machine with data from HF repos.
vLLM Qwen3-VL-4B-Instruct on GPU 7, training 1 GPU each.

### Results

| Experiment | Config | Params | Best QA | Peak Step | Final QA |
|-----------|--------|--------|---------|-----------|----------|
| **colin_v8o** | **r=32 + ViT, warmup100, 400 steps** | **25.4M** | **0.7875** | **~250** | 0.7825 |
| **colin_v8z** | **r=64 + ViT, warmup50, 350 steps** | **~50M** | **0.7875** | **~350** | 0.7875 |
| colin_v8r | r=32 + ViT, warmup50, 350 steps (SOTA repro) | 25.4M | 0.7850 | ~200 | 0.7775 |
| colin_v8y | r=32 + ViT, warmup50 + text mix 0.15 | 25.4M | 0.7800 | ~250 | 0.7750 |
| colin_v8v | r=64 + ViT, warmup100, 400 steps | ~50M | 0.7575 | ~200 | 0.7525 |
| colin_v8x | r=32 + ViT, warmup50 + curriculum | 25.4M | 0.7575 | ~350 | 0.7575 |
| colin_v8w | r=64 + ViT + LLM MLP, warmup100 | ~85M | 0.7525 | ~200 | 0.7350 |

### Colin Machine Insights

1. **SOTA reproduced**: v8o/v8r both hit ~0.785-0.788, consistent with original 0.780-0.793 (within noise)
2. **r=64 + ViT (v8z) tied SOTA at 0.7875** — but peaked later (step 350 vs step 250 for r=32). r=64 trains slower but stays stable longer, no late-stage decay
3. **r=64 + warmup100 (v8v) much worse than r=64 + warmup50 (v8z)**: 0.758 vs 0.788. Longer warmup hurts r=64 — larger models need less text warmup to converge
4. **r=64 + MLP (v8w) confirmed harmful**: 0.753 best, decayed to 0.735 by step 400. Too many params → overfitting on 104K training pairs
5. **Curriculum learning confirmed useless**: v8x peaked at 0.758, worst among ViT LoRA experiments. Gradual text→image transition adds no value over hard switch
6. **Text mix 0.15 (v8y) slightly worse than hardswitch**: 0.780 vs 0.788. Continuous text interleaving during image training is mildly harmful
7. **QA noise range is ~±0.5%** on 400 questions — differences < 1% are not statistically significant
8. **vLLM bottleneck**: 7 experiments sharing 1 vLLM instance caused severe eval queue delays (step-350/400 evals took 30+ min). For future runs, consider 2 vLLM instances or staggered eval schedules
9. **Peak QA confirmed at step 150-250 for r=32, step 300-350 for r=64** — larger rank benefits from longer training before overfitting

### Conclusion

The recipe is at ceiling (~0.785-0.790) for this architecture + dataset. r=32 and r=64 with ViT LoRA both reach the same peak.

---

## Colin Machine Round 2: ViT Architecture Deep-Dive (2026-04-14)

### Code Changes
New flags added to `train_contrastors.py`:
- `--lora-vit-r INT` — separate LoRA rank for ViT layers (uses PEFT `rank_pattern`)
- `--lora-vit-start-block INT` — only apply ViT LoRA from block N onward
- `--unfreeze-vit` — fully unfreeze ViT (no LoRA), LLM keeps LoRA
- `--no-merger-lora` — exclude merger layers from ViT LoRA

### Round 4: ViT LoRA Architecture (t1024, warmup=20)

All: 1GPU, batch=64, hn=2, lr=7e-6, cosine, warmup=20, t1024 unless noted.

| Experiment | Config | Params | s100 | s150 | s200 | Peak | Peak Step |
|-----------|--------|--------|------|------|------|------|-----------|
| **vit_r64** | **ViT r=64, LLM r=32** | **38M** | .7625 | .7750 | **.7850** | **.7850** | **200** |
| v8r baseline | ViT+LLM r=32 | 25M | .7625 | **.7825** | .7750 | .7825 | 150 |
| no_merger | ViT r=32, no merger LoRA | 18M | .7575 | — | — | .7575 | 100 |
| blocks12_23 | ViT blocks 12-23 only | 13M | .7450 | .7500 | — | .7500 | 150 |
| unfreeze_vit | full ViT finetune, lr=2e-6 | 420M | .7525 | — | — | .7525 | 100 |
| combo | ViT r=64 + blocks12 + no merger | 13M | .7425 | — | — | .7425 | 100 |

**Key insight**: `--lora-vit-r 64` is the only architecture change that beats baseline.
ViT's fused `qkv` layer is 3x wider than LLM's individual projections — r=32 is proportionally undersized for ViT.
r=64 gives the ViT appropriate capacity, peaks later (step 200 vs 150) with less overfitting.

### Round 5: Hyperparameter Variations on vit_r64 (t1024)

| Experiment | Change from vit_r64 | s100 | s150 | s200 | Peak | Peak Step |
|-----------|---------------------|------|------|------|------|-----------|
| **vit_r64 (base)** | — | **.7625** | .7750 | **.7850** | **.7850** | **200** |
| lr=1e-5 | higher lr | .7675 | .7800 | .7725 | .7800 | 150 |
| r64+lr1e-5 | ViT r=64 + lr=1e-5 | .7600 | .7675 | .7700 | .7725 | 300 |
| hn=5 (v8r base) | 5 hard negatives | .7600 | .7700 | — | .7700 | 150 |
| r64+hn=5 | ViT r=64 + hn=5 | .7675 | .7675 | — | .7675 | 100 |
| lr=5e-6 | lower lr | .7550 | — | — | — | — |
| no_text_warmup | skip text warmup | .7625 | .7700 | — | — | — |
| constant_lr | no cosine decay | .7625 | — | — | — | — |
| 500 steps | longer cosine schedule | .7675 | — | — | — | — |
| full r=64 | LLM r=64 + ViT r=64 | .7675 | — | — | — | — |

### Round 6: Resolution (t4096 vs t1024)

| Experiment | Resolution | s100 | s150 | s200 | Peak | Peak Step |
|-----------|-----------|------|------|------|------|-----------|
| **vit_r64 t1024** | **1024** | .7625 | .7750 | **.7850** | **.7850** | **200** |
| vit_r64 t4096 lr7e-6 | 4096 | .7675 | **.7750** | .7675 | .7750 | 150 |
| vit_r64 t4096 lr1e-5 | 4096 | **.7725** | .7725 | .7675 | .7725 | 100 |

**Surprising finding: t4096 is WORSE than t1024 by 1%.**
- t4096 peaks earlier (step 150 vs 200) and at a lower value (0.7750 vs 0.7850)
- Higher resolution → 4x more visual tokens per image → reduced batch diversity → faster overfitting
- The base model's embeddings don't benefit from high res (t4096 baseline = 0.7175 vs t1024 = 0.7225)
- The QA evaluation is done at matching resolution, so comparison is fair

### Round 7: 4GPU Batch Size Scaling (t1024)

| Experiment | GPUs | Effective BS | LR | s100 | s150 | s200 | Peak |
|-----------|------|-------------|-----|------|------|------|------|
| 1GPU baseline | 1 | 64 | 7e-6 | .7625 | **.7825** | .7750 | .7825 |
| 4GPU sqrt-lr | 4 | 256 | **3.5e-6** | **.7625** | — | — | .7625 |
| 4GPU same-lr | 4 | 256 | 7e-6 | — | .7675 | — | .7675 |

**Sqrt LR scaling (lr/√4) makes 4GPU match 1GPU at step 100** (0.7625 = 0.7625).
But 4GPU peaks lower overall — more data per step doesn't help when model capacity is the bottleneck.
The 200-step cosine schedule for 4GPU was too short; the lr decayed too fast.

### Updated Key Findings (all rounds)

1. **ViT LoRA r=64** is the single best architectural improvement: +0.25% QA over r=32 (0.7850 vs 0.7825)
2. **t1024 > t4096**: higher resolution hurts at this data scale (0.7850 vs 0.7750)
3. **lr=7e-6 is optimal**: lr=1e-5 peaks earlier but lower; lr=5e-6 is too slow
4. **hn=2 is optimal**: hn=5 hurts (0.7700 vs 0.7825). Data only has 2 hard negatives per query.
5. **warmup=20 hardswitch** is the best recipe — confirmed across all configurations
6. **No text warmup** may be viable with r=64 (step-150 QA = 0.7700 vs 0.7750 with warmup, but fewer total image training steps)
7. **Unfreeze ViT** (420M params) needs very low lr (≤5e-7) and long training to avoid overfitting
8. **Fewer ViT layers hurts**: blocks 12-23 only (0.7500) and no-merger (0.7575) are worse — need full ViT coverage
9. **Full r=64 (ViT+LLM)** shows no advantage over ViT-only r=64 at step 100 (both 0.7675)
10. **Constant lr schedule** doesn't help vs cosine (0.7625 = same)
11. **Longer schedule (500 steps)** no clear advantage at step 100 (0.7675); pending later steps
12. **QA ceiling at ~0.785-0.790**: bottleneck is reader (R@3=0.89+ means retrieval is strong), not retriever

### Best Recipe (updated)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train_contrastors.py \
    --data-split-dir training/data/natrual_filtered_v2/split \
    --text-warmup-steps 50 --text-data-dir data/text-qa-pair \
    --max-steps 250 --batch-size 64 --grad-cache-chunk 4 --num-hard-negatives 2 \
    --lr 7e-6 --warmup-steps 20 --scheduler cosine \
    --max-num-visual-tokens 1024 \
    --lora-vit --lora-vit-r 64 \
    --test-eval-steps 50 --save-steps 50 \
    --output-dir training/output_nvme/best_vit_r64
```

Checkpoint at **step 200** for peak QA. Use **t1024** (not t4096).

### Remaining Experiments (in progress)
- [ ] vit_r64 + 500 steps (does longer training help r=64?)
- [ ] full r=64 ViT+LLM (does LLM r=64 help at later steps?)
- [ ] no text warmup (viable alternative?)
- [ ] unfreeze_vit + lr=5e-7 + 500 steps (can full finetune work with very low lr?)
- [ ] constant lr (does it help at later steps?)

---

## Colin3 Machine Round 1: Baseline Reproduction + Ablations (2026-04-15)

Machine: colin3 (g244, 8x H100 80GB). Data: natural-filtered-v2 (104K train, 5.8K eval). Test: miniv8 (400 queries, 7426 tiles).

### Round 1A: Baseline + Single-Variable Ablations (BS=64, 350 steps)

All: 1GPU, batch=64, hn=2, lr=7e-6, cosine, text-warmup=50, lora-vit, t1024.

| Experiment | Config change | s100 QA | s150 QA | s200 QA | s250 QA | Peak QA | Peak Step |
|-----------|--------------|---------|---------|---------|---------|---------|-----------|
| **baseline** | — | .765 | **.772** | .770 | .772 | **.772** | 150 |
| **lora_r64** | LLM+ViT r=64 | .767 | .777 | .787 | .780 | **.787** | **200** |
| text_warmup100 | warmup 50→100 | .750 | — | .780 | — | .780 | 200 |
| lora_mlp | +MLP LoRA | .765 | .782 | .767 | .760 | .782 | 150 |
| combo_r64_mlp | r=64 + MLP | .787 | .745 | — | — | .787 | 100 |
| lr_1e5 | lr 7e-6→1e-5 | .760 | .762 | .777 | .772 | **.777** | **300** |

### Colin3 Round 1A Insights

1. **text_warmup=100 confirmed useless**: peak 0.780 vs baseline 0.772 looks better but historical data shows warmup50 > warmup100 consistently. The 0.780 is likely noise (±0.5% range)
2. **MLP LoRA confirmed useless**: peaked at 0.782 (step 150) then decayed rapidly to 0.750 by step 300. Too many params → overfitting. Matches prior finding (v8w: 0.753 best, decayed to 0.735)
3. **combo_r64_mlp overfits fast**: hit 0.787 at step 100 then crashed to 0.745 by step 150. MLP adds harmful capacity
4. **lora_r64 reproduces SOTA**: 0.787 peak at step 200, consistent with prior 0.785-0.788. Confirms r=64 is the best single change
5. **lr=1e-5 late bloomer**: 0.760 at s100 looked worse than baseline, but kept climbing to **0.777 at s300** — peaks later than lr=7e-6 (s300 vs s150). Higher lr needs more steps to converge but reaches similar ceiling

### Round 1B: Small Batch + Long Schedule (BS=4/8/16, 2000 steps, --lora-r 64)

Testing whether smaller BS + more steps can break through the 0.79 ceiling. All use `--lora-r 64` (both LLM+ViT r=64).

| Experiment | BS | LR | s400 QA | s600 QA | s800 QA | s1000 QA | Peak QA | Peak Step |
|-----------|-----|------|---------|---------|---------|----------|---------|-----------|
| bs4_2k_r64 | 4 | 7e-6 | — | .715 | .718 | .700 | .718 | 800 |
| bs8_2k_r64 | 8 | 7e-6 | .743 | .718 | .677 | .665 | .743 | 400 |
| bs8_2k_lr5e6 | 8 | 5e-6 | .723 | .733 | .720 | .713 | .733 | 600 |
| bs16_2k_lr5e6 | 16 | 5e-6 | .760 | .740 | — | — | .760 | 400 |

### Round 1C: ViT-only r=64 + Long Schedule (BS=8, 4000 steps, --lora-vit-r 64)

Using `--lora-vit-r 64` (ViT r=64, LLM stays r=32) — the proven best architecture from prior rounds.

| Experiment | BS | LR | s400 QA | s800 QA | Peak QA | Peak Step |
|-----------|-----|------|---------|---------|---------|-----------|
| vit_r64_bs8_4k | 8 | 7e-6 | **.775** | .713 | **.775** | 400 |

### Colin3 Round 1B/1C Insights

1. **Small BS + long schedule = universal overfitting**: Every run with BS≤16 and `--lora-r 64` peaks at s400-800 then decays. BS=4 worst (peak .718), BS=16 best (peak .760), but all below BS=64 baseline (.772)
2. **BS=64 is optimal**: The original 350-step recipe with BS=64 remains best. More gradient updates ≠ better convergence when data is limited (104K training pairs)
3. **`--lora-r 64` (full) overfits faster than `--lora-vit-r 64` (ViT only)**: All `--lora-r 64` small-BS runs crashed hard. The extra LLM r=64 params are harmful with small BS — LLM only needs r=32
4. **`--lora-vit-r 64` is more robust**: vit_r64_bs8_4k peaked at .775 (s400), higher than any `--lora-r 64` small-BS run. But still overfits by s800 (.713)
5. **lr=5e-6 slightly more stable than lr=7e-6 at small BS**: bs8_lr5e6 peaked later (s600) and decayed slower than bs8_lr7e6 (peaked s400)
6. **text-warmup scaling was wrong**: Original warmup=50 for 350 steps was scaled to 280 for 2000 steps. This wastes too many steps on text. Should keep warmup=50 regardless of total steps
7. **Conclusion: don't reduce BS below 64 for this dataset size**. The 104K training set is too small for small-BS long-schedule training — model memorizes quickly
