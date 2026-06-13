# VLM evaluation — SLURM jobs

Three env-driven jobs cover the whole VLM evaluation, in order:

| Step | Job | Partition | What it does |
|------|-----|-----------|--------------|
| 1 | `build_dataset.sub` | CPU (`defq`) | Build the 2-D eval dataset (slices + overlays + ground truth) |
| 2 | `run_vlm.sub` | GPU (`vram48`) | Run one model across all 4 conditions × 3 tasks |
| 3 | `analyze.sub` | CPU (`defq`) | Tables, figures, stats, `report.md` (the `eval_analysis` engine) |

All jobs source `infra/slurm/env.sh` (so `PROJ_DIR`, `ITRIALSPACE_DATA_DIR`,
`ITRIALSPACE_OUTPUT_DIR`, conda, partitions, and `.env` are resolved automatically) and are
configured entirely through environment variables — **no editing**. Submit from the repo root.

Models and profiles: **BiomedCLIP** and **LLaVA-Med** use the `lung_axial` eval set;
**MedGemma** uses `lung_axial_medgemma` (3-channel CT). Build both profiles to evaluate all three.

---

## Quickstart — the small DEMO (defaults)

Uses the demo data under `$ITRIALSPACE_OUTPUT_DIR` (e.g.
`/scratch/railabs/ft42/VLST_Project/Data/outputs_iTrialSpace`). Everything below is copy-paste.

```bash
cd <repo root>

# 1) Build demo eval sets (synthetic modes 1–3, both profiles; + a small real set)
sbatch infra/slurm/vlm_eval/build_dataset.sub                               # synthetic · lung_axial
PROFILE=lung_axial_medgemma sbatch infra/slurm/vlm_eval/build_dataset.sub   # synthetic · medgemma
VLM_SET=real sbatch infra/slurm/vlm_eval/build_dataset.sub                  # real · lung_axial
VLM_SET=real PROFILE=lung_axial_medgemma sbatch infra/slurm/vlm_eval/build_dataset.sub

# 2) Run the three models on the synthetic demo
B=$ITRIALSPACE_OUTPUT_DIR/vlm_eval_demo
MODEL=biomedclip EVAL_DIR=$B/lung_axial          sbatch infra/slurm/vlm_eval/run_vlm.sub
MODEL=llava_med  EVAL_DIR=$B/lung_axial          sbatch infra/slurm/vlm_eval/run_vlm.sub
MODEL=medgemma   EVAL_DIR=$B/lung_axial_medgemma sbatch infra/slurm/vlm_eval/run_vlm.sub

# 3) Analyse everything found under the demo output
sbatch infra/slurm/vlm_eval/analyze.sub          # → $B/eval_analysis/report.md
```

Chain steps with SLURM dependencies if you want one submission:
```bash
J=$(sbatch --parsable infra/slurm/vlm_eval/build_dataset.sub)
R=$(MODEL=biomedclip sbatch --parsable --dependency=afterok:$J infra/slurm/vlm_eval/run_vlm.sub)
sbatch --dependency=afterok:$R infra/slurm/vlm_eval/analyze.sub
```

---

## Scaling to the full / curated dataset

The released dataset already ships the **built** eval sets and **frozen splits** under
`$ITRIALSPACE_DATA_DIR/vlm_dataset/{synthetic,real}/<profile>/` and
`…/vlm_dataset/splits/` — so on the shared data you usually **skip step 1** and go straight to
step 2, scoring a fixed split:

```bash
B=$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic
S=$ITRIALSPACE_DATA_DIR/vlm_dataset/splits/release_v1_full.synthetic.txt
MODEL=biomedclip CASE_IDS=$S EVAL_DIR=$B/lung_axial            sbatch infra/slurm/vlm_eval/run_vlm.sub
MODEL=llava_med  CASE_IDS=$S EVAL_DIR=$B/lung_axial            sbatch infra/slurm/vlm_eval/run_vlm.sub
MODEL=medgemma   CASE_IDS=$S EVAL_DIR=$B/lung_axial_medgemma   sbatch infra/slurm/vlm_eval/run_vlm.sub
# (same for the real set under …/vlm_dataset/real with release_v1_full.real.txt)

RESULTS=$ITRIALSPACE_DATA_DIR/vlm_dataset SPLIT=release_v1_full \
  sbatch infra/slurm/vlm_eval/analyze.sub
```

To **regenerate** the full eval sets from scratch (step 1 on all data):
```bash
# synthetic, all 13 modes, both profiles
VLM_MODES="1 2 3 4 5 6 7 8 9 10 11 12 13" \
  EVAL_DIR=$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic/lung_axial \
  sbatch infra/slurm/vlm_eval/build_dataset.sub
VLM_MODES="1 2 3 4 5 6 7 8 9 10 11 12 13" PROFILE=lung_axial_medgemma \
  EVAL_DIR=$ITRIALSPACE_DATA_DIR/vlm_dataset/synthetic/lung_axial_medgemma \
  sbatch infra/slurm/vlm_eval/build_dataset.sub
# real, all 7 datasets, ALL cases (VLM_MAX="")
VLM_SET=real VLM_MAX="" VLM_DATASETS="DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT" \
  EVAL_DIR=$ITRIALSPACE_DATA_DIR/vlm_dataset/real/lung_axial \
  sbatch infra/slurm/vlm_eval/build_dataset.sub
# (+ the lung_axial_medgemma profile for real)
```

## Transferring to another cluster

Set these (in `.env` or the submit environment) and nothing else changes:
`ITRIALSPACE_DATA_DIR`, `ITRIALSPACE_OUTPUT_DIR`, `ITRIALSPACE_CONDA_ENV`, and the partitions
`SLURM_PARTITION_GPU` / `SLURM_PARTITION_CPU`. The `#SBATCH --partition` defaults here are
`vram48` (GPU) / `defq` (CPU); override per submit with `sbatch -p <partition> …`. Gated models
(MedGemma) read `HF_TOKEN` from `.env` (see `.env.example`).

## Env knobs (summary)

| Job | Knobs |
|-----|-------|
| `build_dataset.sub` | `VLM_SET` · `PROFILE` · `EVAL_DIR` · synthetic: `VLM_MODES`, `OUT_BASE` · real: `VLM_DATASETS`, `VLM_MAX`, `DATA_BASE` |
| `run_vlm.sub` | `MODEL` · `EVAL_DIR` · `CASE_IDS` · `CONDITIONS` |
| `analyze.sub` | `RESULTS` · `SPLIT` · `OUT` · `NBOOT` |

See [docs/vlm_eval.md](../../../docs/vlm_eval.md) for the full guide and
[docs/vlm_eval_implementation.md](../../../docs/vlm_eval_implementation.md) for the architecture.
