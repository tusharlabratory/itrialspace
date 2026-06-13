# infra/bash — run iTrialSpace on a single machine (Docker / bash, no SLURM)

First-class, scheduler-free drivers for one machine (a workstation or a single GPU server). This is
the **primary** path for Docker hosts; it is the peer of [`infra/slurm/`](../slurm/) (for clusters),
not a shim over it. Both read the **same `.env`** and run the **same per-mode logic** — only the
launcher differs (a bash loop here vs. `sbatch --array` there), so outputs are identical.

## Files

| Path | Purpose |
|------|---------|
| `env.sh` | Shared environment: loads `.env`, sets `PYTHONPATH`, picks a **writable** log dir, makes conda **optional** (silent when the package already imports). Sourced by every stage script. |
| `run_pipeline.sh` | Core pipeline for one mode: **trials → insert → synth**, sequentially, with per-run logging. |
| `run_sweep.sh` | Run one stage across many modes **concurrently**, with a `JOBS` cap (the analogue of submitting many SLURM jobs). GPU-aware for synth. |
| `trials/mode<1-13>_*.sh` | Dedicated per-mode trial scripts (cohort manifests). Clean bash, no `#SBATCH`. |
| `mask_inserter/insert_masks.sh` | Insert donor masks (label 23) into host anatomy. One generic mode-driven script (the SLURM per-mode files differ only by name). Manifest-level parallel via `JOBS`. |
| `synthesis/synthesize.sh` | NodMAISI CT synthesis with **case-level GPU parallelism** (the bash equivalent of the SLURM synth array). Spreads a mode's cases across GPUs. |
| `vlm_eval/{build_dataset,run_vlm,analyze}.sh` | VLM evaluation, 3 steps: build the 2-D eval set → run a model (GPU) → analyze → `report.md`. |
| `nodulemap/{build,serve}.sh` | NoduleMap app: build artifacts (embeddings + KNN edges) then serve the explorer (port 8422). |
| `retriever/{serve,cli}.sh` | Retriever app: serve FastAPI API + Streamlit UI (8421/8501), or the CLI (search/similar/match/…). |
| `docker_run.sh` | Run any command in the GPU image with `--gpus` / `--user` / `--shm-size` / `HF_HOME` cache / mounts / `.env` wired up. |
| `docker_save.sh` | Package the built `itrialspace:gpu` image into a portable tarball for sharing. |

## 1. Build the image (once)

```bash
docker build -t itrialspace:gpu -f docker/Dockerfile.gpu .
```

Needs the NVIDIA Container Toolkit (`--gpus all`). The image is **code-only** — data, weights and
outputs are mounted at runtime and pointed to by `.env`.

## 2. Configure paths + token (once)

Everything host-specific lives in the gitignored `.env` at the repo root (see `.env.example`):

```bash
ITRIALSPACE_DATA_DIR=/path/to/iTrialSpace        # inputs: profiles/ masks/ (raw_ct/ optional)
ITRIALSPACE_OUTPUT_DIR=/path/to/outputs          # manifests/ inserted_masks/ generated_cts/ logs/
NODMAISI_MODELS_DIR=/path/to/nodmaisi/models     # synthesis weights (Step 3 only)
HF_TOKEN=hf_xxx                                   # gated VLMs (MedGemma) only
```

`docker_run.sh` reads these automatically — no need to export anything.

## 3. Run the pipeline

```bash
# host conda install:
infra/bash/run_pipeline.sh <mode 1-13> [all|trials|insert|synth]

# inside the GPU image (the wrapper sets --gpus/--user/mounts and reads .env):
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/run_pipeline.sh 1'
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/run_pipeline.sh 1 trials'
ENTRY=bash infra/bash/docker_run.sh -lc 'for m in $(seq 1 13); do bash infra/bash/run_pipeline.sh $m trials; done'
```

- `all` (default) = `trials → insert → synth`.
- **trials / insert** are CPU; **synth** needs a visible GPU + `NODMAISI_MODELS_DIR`.
- One failing case is logged as a `WARN` and the run continues.
- Output → `$ITRIALSPACE_OUTPUT_DIR/{manifests,inserted_masks,generated_cts}/mode<MODE>/…`.

### End-to-end (one mode, then all 13)

```bash
W() { ENTRY=bash infra/bash/docker_run.sh -lc "$1"; }   # tiny helper for the examples below

# one mode, full core pipeline (trials → insert → synth), synth packed 4 cases/GPU:
W 'PER_GPU=4 bash infra/bash/run_pipeline.sh 1 all'

# all 13 modes, stage by stage (recommended order on a single box):
W 'for m in $(seq 1 13); do bash infra/bash/run_pipeline.sh $m trials; done'   # CPU
W 'for m in $(seq 1 13); do JOBS=4 bash infra/bash/mask_inserter/insert_masks.sh $m; done'  # CPU, manifests parallel
W 'for m in $(seq 1 13); do PER_GPU=4 bash infra/bash/run_pipeline.sh $m synth; done'        # GPU

# then VLM (build → run each model → analyze) — see §6.
```

## Trial modes (what `<mode>` means)

`<mode>` is `1`–`13`; each is a different virtual-trial design. The bash script, the runner's
`--mode` value, and the question each answers:

| # | bash script (`infra/bash/trials/`) | `--mode` | Question it answers |
|---|------------------------------------|----------|---------------------|
| 1 | `mode1_controlled_prevalence.sh`     | `controlled_prevalence` | Performance vs. cancer **prevalence** |
| 2 | `mode2_size_detection_curve.sh`      | `size_detection_curve`  | Detection sensitivity vs. nodule **size** (FROC) |
| 3 | `mode3_location_sensitivity.sh`      | `location_sensitivity`  | Detection vs. anatomical **lobe** |
| 4 | `mode4_demographic_stratification.sh`| `demographic_strat`     | Performance across **demographic** strata |
| 5 | `mode5_counterfactual.sh`            | `counterfactual`        | Same cohort swept over one parameter (e.g. prevalence) |
| 6 | `mode6_cross_dataset.sh`             | `cross_dataset`         | Generalization across acquisition **sources** |
| 7 | `mode7_bootstrap_confidence.sh`      | `bootstrap_confidence`  | **Confidence intervals** via resampled cohorts |
| 8 | `mode8_algorithm_comparison.sh`      | `algorithm_comparison`  | Standardized head-to-head model comparison |
| 9 | `mode9_screening_simulation.sh`      | `screening_simulation`  | Multi-round screening with **prevalence decay** |
| 10 | `mode10_multi_nodule_realism.sh`    | `multi_nodule`          | Impact of **companion** nodules in context |
| 11 | `mode11_digital_twin_isolation.sh`  | `digital_twin_isolation`| Each native nodule **isolated** in its own host anatomy |
| 12 | `mode12_digital_twin_complete.sh`   | `digital_twin_complete` | **All** native nodules of a scan, reconstructed |
| 13 | `mode13_digital_twin_cross.sh`      | `digital_twin_cross`    | Donor nodules placed in **cross-patient** host anatomy |

Per-mode parameters (and demo vs. paper sizes) are in **[§ Run size](#run-size--demo--paperproduction)**;
full design notes in [docs/trial_modes.md](../../docs/trial_modes.md).

### Run size — demo → paper/production

The **trials** scripts ship a deliberately small **demo** (≈5 cases/mode) so a full sweep finishes in
seconds. Only the **trials** stage has a "run size" — insert and synth automatically process whatever
trials produced (scale their *speed* with `JOBS` / `N_JOBS` / `PER_GPU`, §3a/§3b).

Per-mode demo vs. paper values (same numbers the SLURM `.sub` files use; 7 datasets =
`DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT`):

| Mode | Knob(s) | Demo | Paper |
|------|---------|------|-------|
| 1 controlled_prevalence   | `N_CASES` | 5 | **1000** |
| 2 size_detection_curve    | `N_PER_BUCKET` | 5 | **100** |
| 3 location_sensitivity    | `N_PER_LOBE` | 5 | **100** |
| 4 demographic_strat       | `N_PER_STRATUM` | 5 | **200** |
| 5 counterfactual          | `N_CASES` (per variant) | 5 | **500** |
| 6 cross_dataset           | `N_CASES` | 5 | **300** |
| 7 bootstrap_confidence    | `N_CASES` / `N_BOOTSTRAP` | 5 / 3 | **200 / 20** |
| 8 algorithm_comparison    | `N_CASES` | 5 | **500** |
| 9 screening_simulation    | `N_CASES_PER_ROUND` | 5 | **500** |
| 10 multi_nodule_realism   | `N_CASES` | 5 | **500** |
| 11 digital_twin_isolation | `DATASETS` / `MAX_PATIENTS` | `DLCS24` / 5 | **all 7 / `all`** |
| 12 digital_twin_complete  | `DATASETS` / `MAX_PATIENTS` | `DLCS24` / 5 | **all 7 / `all`** |
| 13 digital_twin_cross     | `HOST_DATASETS` / `DONOR_DATASETS` / `MAX_DONOR_NODULES` | `DLCS24` / `LUNA25` / 5 | **all 7 / all 7 / 250** |

**Two ways to switch to paper size** (pick one):

**(a) Per-run env override** — nothing to edit (the bash advantage over SLURM; set the var(s) for that mode):
```bash
ENTRY=bash infra/bash/docker_run.sh -lc 'N_CASES=1000 bash infra/bash/run_pipeline.sh 1 trials'
ENTRY=bash infra/bash/docker_run.sh -lc 'N_CASES=200 N_BOOTSTRAP=20 bash infra/bash/run_pipeline.sh 7 trials'
# modes 11/12 at full coverage:
ENTRY=bash infra/bash/docker_run.sh -lc 'DATASETS="DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT" MAX_PATIENTS=all bash infra/bash/run_pipeline.sh 11 trials'
# mode 13 all host×donor pairs:
ENTRY=bash infra/bash/docker_run.sh -lc 'HOST_DATASETS="DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT" DONOR_DATASETS="DLCS24 LUNA25 LUNA16 LUNGx LNDbv4 NSCLCR IMDCT" MAX_DONOR_NODULES=250 bash infra/bash/run_pipeline.sh 13 trials'
```

**(b) Permanent** — edit the default in `infra/bash/trials/mode<N>_*.sh` (change the `${VAR:-5}`
default to the paper value), exactly analogous to editing the `#SBATCH`/value line in the SLURM `.sub`.
Then a plain `run_pipeline.sh <N> trials` uses the paper size.

After trials, run insert and synth as usual — they pick up the larger cohorts automatically; just
raise their parallelism (`JOBS`, `PER_GPU`) for the bigger workload (§3a, §3b).

## 3a. Insertion (CPU) — single mode / all modes / parallel

Insert donor nodule masks (label 23) into host anatomy from the trial manifests. Reads
`manifests/mode<N>_*/…`, writes `inserted_masks/mode<N>_*/…/{*_mask.nii.gz, audit.json}`.

```bash
# SINGLE mode:
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/run_pipeline.sh 1 insert'

# ALL 13 modes (sequential):
ENTRY=bash infra/bash/docker_run.sh -lc 'for m in $(seq 1 13); do bash infra/bash/run_pipeline.sh $m insert; done'
```

**Two parallelism dials** (CPU):

| Dial | Where | Default | Effect |
|------|-------|---------|--------|
| `N_JOBS` | any insert call | 8 | worker processes across cases inside one manifest (SLURM `--n-jobs`) |
| `JOBS`   | `insert_masks.sh` | 1 | manifests of one mode run at once (SLURM "array over manifests") |
| `JOBS`   | `run_sweep.sh insert` | 4 | whole modes run at once |

```bash
# manifests of mode 2 in parallel (6 manifests × 8 workers ≈ 48 cases in flight):
ENTRY=bash infra/bash/docker_run.sh -lc 'JOBS=6 bash infra/bash/mask_inserter/insert_masks.sh 2'

# all 13 modes, 4 modes at a time:
ENTRY=bash infra/bash/docker_run.sh -lc 'JOBS=4 bash infra/bash/run_sweep.sh insert'
```

> In-flight CPU workers ≈ (modes in parallel) × (manifests in parallel) × `N_JOBS`. Keep that product
> ≤ free cores (`nproc`). Demo cases are seconds each — leave defaults; dial up only for paper-size
> runs on an idle box.

## 3b. Synthesis (GPU) — single mode / all modes / parallel

NodMAISI CT synthesis from the insertion audits. Needs a GPU + `NODMAISI_MODELS_DIR`. Reads
`inserted_masks/…/audit.json`, writes `generated_cts/mode<N>/<manifest>/<case>/synthetic_ct.nii.gz`.
**A case needs ~20 GB GPU memory** (full `[80,80,64]` sliding window).

```bash
# SINGLE mode — cases spread across ALL visible GPUs (case-level parallel, the default):
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/run_pipeline.sh 1 synth'

# ALL 13 modes (each mode uses all GPUs in turn):
ENTRY=bash infra/bash/docker_run.sh -lc 'for m in $(seq 1 13); do bash infra/bash/run_pipeline.sh $m synth; done'
```

**Parallelism dials** (GPU):

| Dial | Where | Default | Effect |
|------|-------|---------|--------|
| (case-level) | `synthesize.sh` | always on | a mode's cases are dispatched across GPUs — the SLURM `--array` over cases |
| `GPUS="0 1 2 3"` | `synthesize.sh` | all visible | which GPUs to use |
| `PER_GPU` | `synthesize.sh` | 1 | cases per GPU at once → `#GPUS × PER_GPU` cases in flight |
| `JOBS` | `run_sweep.sh synth` | #GPUs | whole modes at once, one mode pinned per GPU |

```bash
# one mode, 4 cases per GPU (4 GPUs × 4 = 16 in flight). H200 143 GB ÷ ~20 GB/case → up to ~4-5/GPU:
ENTRY=bash infra/bash/docker_run.sh -lc 'PER_GPU=4 bash infra/bash/run_pipeline.sh 1 synth'

# all 13 modes, 4 cases per GPU:
ENTRY=bash infra/bash/docker_run.sh -lc 'for m in $(seq 1 13); do PER_GPU=4 bash infra/bash/run_pipeline.sh $m synth; done'

# alternative: run modes concurrently, one mode pinned to each GPU (good for many small modes):
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/run_sweep.sh synth'
```

> **Per-mode loop vs. `run_sweep synth`** — both saturate your GPUs; pick by shape: the loop gives all
> GPUs to one mode's cases (best when modes have many cases); `run_sweep synth` runs N modes at once,
> one GPU each (best for many small modes). Don't combine `run_sweep synth` with `PER_GPU` (it would
> oversubscribe). Memory guard: synthesis needs the full GPU — on a shared card it OOM-retries with a
> smaller window (different output); ensure ~20 GB free per concurrent case so the full window is used.

## 4. Logs

Every run is teed to **terminal + file**:

```
$ITRIALSPACE_OUTPUT_DIR/logs/mode<MODE>_<stage>_<YYYYMMDD_HHMMSS>.log
```

The driver prints the exact path at the end (`log: …`). Override with `ITS_LOG_DIR=/some/dir`, or
disable file logging with `ITS_NO_LOG=1`. (Logs go to the output dir, **not** the repo, because the
image's code dir `/app` is read-only at runtime — only mounted volumes are writable.)

## 5. The wrapper — `docker_run.sh` knobs

| Env | Effect |
|-----|--------|
| *(none)* | paths + `HF_TOKEN` read from `.env`; runs as your host uid; mounts data/out/(models) |
| `ENTRY=bash` / `ENTRY=python` / `ENTRY=its-nodulemap` | override the entrypoint (default is the `its` CLI) |
| `ITS_DEV=1` | mount the host repo over `/app` so script edits take effect **without rebuilding** |
| `ITS_PORTS="8422 8421 8501"` | publish app ports |
| `ITS_DATA` / `ITS_OUT` / `ITS_MODELS` | override the `.env` paths for one run |

```bash
ITS_DATA=/host/iTrialSpace infra/bash/docker_run.sh config      # `its config`
ENTRY=bash infra/bash/docker_run.sh -lc 'pytest -m "not full_volume"'
```

## 6. VLM evaluation (build → run → analyze)

Three steps (clean bash scripts under `infra/bash/vlm_eval/`, mirroring the SLURM `.sub` files):

```bash
# 1) BUILD the 2-D eval dataset (CPU). Demo = synthetic modes 1 2 3, profile lung_axial.
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/vlm_eval/build_dataset.sh'

# 2) RUN a model across all 4 conditions × 3 tasks (GPU). biomedclip/llava_med are open;
#    medgemma is gated (needs HF_TOKEN + the lung_axial_medgemma eval set).
ENTRY=bash infra/bash/docker_run.sh -lc 'MODEL=biomedclip GPU=0 bash infra/bash/vlm_eval/run_vlm.sh'

# 3) ANALYZE → tables, figures, McNemar significance, report.md (CPU).
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/vlm_eval/analyze.sh'
```

Knobs: build — `VLM_SET` (synthetic|real), `PROFILE` (lung_axial|lung_axial_medgemma), `VLM_MODES`,
`EVAL_DIR`; run — `MODEL`, `EVAL_DIR`, `CONDITIONS`, `GPU`; analyze — `RESULTS`, `SPLIT`, `NBOOT`.
**Run several models in parallel**, one per GPU:

```bash
ENTRY=bash infra/bash/docker_run.sh -lc 'MODEL=biomedclip GPU=0 bash infra/bash/vlm_eval/run_vlm.sh &
                                         MODEL=llava_med  GPU=1 bash infra/bash/vlm_eval/run_vlm.sh & wait'
```

MedGemma needs its own eval set built first: `PROFILE=lung_axial_medgemma … build_dataset.sh`, then
`MODEL=medgemma EVAL_DIR=…/lung_axial_medgemma … run_vlm.sh`.

> The image pins the VLM stack to the dev-env versions (`transformers==4.51.3`, `open_clip_torch==2.32.0`,
> …) — the unpinned latest `transformers` breaks on torch 2.6. Model downloads are cached to
> `$ITRIALSPACE_OUTPUT_DIR/.hf_cache` (via `HF_HOME`) so they download once.

## 6a. Apps — NoduleMap & Retriever

Interactive apps over the nodule space. In Docker, **publish the ports** with `ITS_PORTS="…"` and open
`http://localhost:<port>` on the host. The wrapper automatically **remaps host paths to container
mounts** (e.g. a `.env` `NODULEMAP_ARTIFACTS=/mnt/…/nodulemap_artifacts` under your data dir becomes
`/data/nodulemap_artifacts` inside the container), so artifacts you built on the host are found without
any extra flags.

### NoduleMap (embedding-graph explorer)

```bash
# 1) build artifacts ONCE (embeddings + KNN edges) → $NODULEMAP_ARTIFACTS
#    (default $ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts). Skip this if you already have them.
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/nodulemap/build.sh'

# 2) serve (port 8422) → open http://localhost:8422
ITS_PORTS="8422" ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/nodulemap/serve.sh'
```

- **Already built?** `serve.sh` loads the existing `$NODULEMAP_ARTIFACTS` directly — no rebuild needed.
  It errors clearly ("build first") if the dir is missing.
- **Knobs:** `NODULEMAP_PORT` (8422), `NODULEMAP_ARTIFACTS` (host path auto-remapped into the container).
- **Ports:** `serve.sh` binds `0.0.0.0:8422` *inside* the container; `ITS_PORTS="8422"` publishes it to
  the host. Without `ITS_PORTS`, the app runs but isn't reachable from your browser.
- **Data:** `build.sh` reads profiles/masks from the mounted `/data`, so `.env`'s `ITRIALSPACE_DATA_DIR`
  must point at real data.
- SLURM: `sbatch infra/slurm/nodulemap/rebuild_nodulemap.sub` then `sbatch infra/slurm/nodulemap/nodulemap.sub`.

### Retriever (faceted search + CT viewer)

```bash
# serve FastAPI API (8421) + Streamlit UI (8501) → open http://localhost:8501
ITS_PORTS="8421 8501" ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/retriever/serve.sh'

# or the CLI (no server): info / search / similar / match / detail / export
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/retriever/cli.sh info'
ENTRY=bash infra/bash/docker_run.sh -lc 'bash infra/bash/retriever/cli.sh search --label 1 --lobe right_lung_upper_lobe --limit 50'
```

- **Knobs:** `RETRIEVER_API_PORT` (8421), `RETRIEVER_UI_PORT` (8501). The UI talks to the API at
  `http://localhost:$RETRIEVER_API_PORT`; `serve.sh` starts both and stops both on Ctrl-C.
- **Ports:** publish **both** with `ITS_PORTS="8421 8501"` (the UI needs the API port reachable too).
- SLURM: `sbatch infra/slurm/retriever/app.sub` (API+UI) · `sbatch infra/slurm/retriever/cli.sub`.

## 7. Share the built image

```bash
infra/bash/docker_save.sh                 # -> itrialspace-gpu.tar.gz  (docker load on the target)
# or push to a registry:  docker tag itrialspace:gpu <reg>/itrialspace:gpu && docker push <reg>/itrialspace:gpu
```

Full single-server guide: [docs/running_on_a_server.md](../../docs/running_on_a_server.md) ·
container/sharing details: [docs/installation.md](../../docs/installation.md#containers).
