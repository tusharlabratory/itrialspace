# iTrialSpace -> NodMAISI Integration Pipeline


---

## 1. Integration Architecture (High-Level)

```
iTrialSpace Manifest         Mask Insertion Engine            NodMAISI
+------------------+    +---------------------------+    +------------------+
| CohortManifest   |--->| insert_manifest()         |--->| Combined masks   |
| (CSV, 9 modes)   |    | - placement               |    | (label 23 =      |
|                  |    | - resample                |    |  nodule, uint8)  |
+------------------+    | - label stamping (23)     |    +--------+---------+
                        +---------------------------+             |
                                                                  v
                        +---------------------------+    +------------------+
                        | run_itrialspace_to_ct.py  |<---| audit.json       |
                        |                           |    +------------------+
                        | Per case:                 |
                        |  1. Load mask             |
                        |  2. Canonicalize RAS+     |
                        |  3. Resize to valid dims  |
                        |  4. Write dataset JSON    |
                        |  5. Run NodMAISI infer    |
                        |  6. Sanity checks         |
                        |  7. QC PNGs               |
                        |  8. Write audit           |
                        +------------+--------------+
                                     |
                                     v
                        +---------------------------+
                        | generated_cts/            |
                        |   mode1_.../              |
                        |     case_0000/            |
                        |       synthetic_ct.nii.gz |
                        |       input_mask.nii.gz   |
                        |       host_ct.nii.gz ->   |
                        |       nodmaisi_audit.json |
                        |       qc/                 |
                        |         qc_slice_*.png    |
                        |     case_0001/            |
                        |     ...                   |
                        |   pipeline_summary.json   |
                        +---------------------------+
```

### Data Flow

| Stage | Input | Output | Tool |
|-------|-------|--------|------|
| 1. Manifest planning | Raw datasets, profiles | CohortManifest CSV | iTrialSpace |
| 2. Mask insertion | Manifest + host organ segs + donor masks | Combined masks (label 23) + audit.json | itrialspace_mask_inserter |
| 3. CT synthesis | Combined masks + audit.json | Synthetic CTs + QC | run_itrialspace_to_ct.py + NodMAISI |

---

## 2. NodMAISI Input Requirements (Determined from Code Analysis)

### 2.1 What NodMAISI Expects

The ControlNet inference script (`infer_testV2_controlnet.py`) requires:

1. **A dataset JSON** with structure:
   ```json
   {"testing": [{"label": "mask.nii.gz", "dim": [512,512,256], "spacing": [0.7,0.7,1.25]}]}
   ```

2. **A segmentation label mask** (NIfTI, uint8) — this is the ControlNet conditioning input.
   - Loaded via MONAI `LoadImaged`, oriented to RAS via `Orientationd(axcodes="RAS")`
   - Converted to 8-channel binary via `binarize_labels()` (bitwise decomposition)
   - The 8-bit binary representation is fed as `controlnet_cond` (shape: `[B, 8, H, W, D]`)

3. **Label 23 = lung nodule/tumor** — this is the same label used by iTrialSpace! No remapping needed.

4. **Spacing** is multiplied by 100 before being fed to the diffusion U-Net as a conditioning tensor.

5. **Modality** = 1 (CT) as an integer class embedding.

### 2.2 Size Constraints (from `check_input()`)

- `dim[0] == dim[1]` (square axial slices), must be in `{256, 384, 512}`
- `dim[2]` must be in `{128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]` (isotropic in-plane)
- The latent shape is `(4, dim[0]//4, dim[1]//4, dim[2]//4)` — all dims must be divisible by 4

### 2.3 Output Format

- Synthetic CT in HU range `[-1000, 1000]`
- Background (where mask == 0) is set to -1000 HU
- Saved as NIfTI via MONAI `SaveImage` with `_image` postfix

---

## 3. Geometry/Orientation Handling

### 3.1 Preprocessing (mask -> NodMAISI input)

The driver script `run_itrialspace_to_ct.py` performs:

1. **Load mask** via nibabel
2. **Canonicalize to RAS+** via `nib.as_closest_canonical()`
3. **Snap dimensions** to nearest valid NodMAISI size:
   - XY: nearest of {256, 384, 512}
   - Z: nearest of {128, 256, 384, 512, 640, 768}
4. **Resize if needed** using `scipy.ndimage.zoom(order=0)` (nearest-neighbor for labels)
5. **Adjust spacing** proportionally: `new_spacing = old_spacing * old_dim / new_dim`
6. **Write** resized mask as uint8 NIfTI with diagonal affine

### 3.2 Post-inference Geometry Check

After NodMAISI inference:
- Verify synthetic CT shape matches the input mask shape
- Verify spacing consistency
- Check for NaN values
- Verify HU range is within [-1000, 1000]
- All recorded in per-case audit JSON

### 3.3 No Back-Resampling Needed

NodMAISI outputs in the same grid as the input mask. If the mask was resized, the synthetic CT is at the resized resolution. The original dimensions are recorded in the audit for downstream consumers that may need to resample back.

---

## 4. Output Directory Structure

```
generated_cts/
└── mode1_controlled_prevalence/
    ├── case_0000/
    │   ├── synthetic_ct.nii.gz        # Generated CT [-1000, 1000] HU
    │   ├── input_mask.nii.gz          # Prepared mask (RAS+, resized)
    │   ├── host_ct.nii.gz             # Symlink to original host CT
    │   ├── nodmaisi_audit.json        # Per-case audit (params, timing, checks)
    │   ├── dataset.json               # NodMAISI dataset JSON (intermediate)
    │   ├── _tmp_env.json              # NodMAISI env config (intermediate)
    │   └── qc/
    │       ├── qc_slice_00_z0118.png  # Axial slice through nodule
    │       ├── qc_slice_01_z0125.png
    │       └── qc_slice_02_z0132.png
    ├── case_0001/
    │   └── ...
    └── pipeline_summary.json          # Global run summary
```

---

## 5. File Naming Conventions

| File | Convention | Source |
|------|-----------|--------|
| Case directory | `case_{case_id:04d}` | iTrialSpace case_id, zero-padded |
| Synthetic CT | `synthetic_ct.nii.gz` | Fixed name in config |
| Input mask | `input_mask.nii.gz` | Preprocessed copy of inserted mask |
| Host CT link | `host_ct.nii.gz` | Symlink to original |
| QC slices | `qc_slice_{idx:02d}_z{slice:04d}.png` | Axial slice index |
| Per-case audit | `nodmaisi_audit.json` | Fixed name in config |

---

## 6. QC Outputs

For each case, when QC is enabled:

### PNG Montages
- 3 axial slices (configurable) through the nodule region
- Each PNG shows side-by-side panels:
  - **Host CT** (if available) at the same slice
  - **Synthetic CT + Mask overlay** (nodule label 23 in red)
  - **Synthetic CT** alone
- Window: center=-600, width=1500 (lung window)

### Audit JSON (per-case)
```json
{
  "case_id": "0042",
  "status": "success",
  "timestamp": "2026-03-06T14:32:15",
  "input_mask_path": "/scratch/.../inserted_masks/.../mask.nii.gz",
  "synthetic_ct_path": "/scratch/.../generated_cts/.../synthetic_ct.nii.gz",
  "geometry": {
    "original_dim": [412, 412, 262],
    "original_spacing": [0.703, 0.703, 1.25],
    "nodmaisi_dim": [384, 384, 256],
    "nodmaisi_spacing": [0.754, 0.754, 1.28],
    "resized": true
  },
  "nodmaisi_params": {
    "num_inference_steps": 30,
    "noise_factor": 1.0,
    "modality": 1,
    "seed": 42
  },
  "timing": {
    "prep_sec": 2.3,
    "inference_sec": 185.7,
    "qc_sec": 1.2,
    "total_sec": 189.2
  },
  "sanity_checks": {
    "shape_match": true,
    "has_nodule_label": true,
    "ct_min_hu": -1000.0,
    "ct_max_hu": 812.3,
    "has_nan": false
  }
}
```

---

## 7. Runnable Commands

### 7.1 Local Test — Single Case

```bash
cd /home/ft42/NoMAISI
module load miniconda/py39_4.12.0
source activate monai-auto3dseg
export MONAI_DATA_DIRECTORY=/home/ft42/NoMAISI/
export PYTHONPATH=/home/ft42/NoMAISI:$PYTHONPATH

# Dry run first (no GPU needed)
python tools/run_itrialspace_to_ct.py \
    --audit /scratch/railabs/ft42/VLST_Project/Data/iTrialSpace/inserted_masks/mode1_controlled_prevalence/audit.json \
    --config tools/integration_config.yaml \
    --outdir /tmp/test_nodmaisi \
    --case-ids 0 \
    --dry-run -vv

# Real inference (requires GPU)
python tools/run_itrialspace_to_ct.py \
    --audit /scratch/railabs/ft42/VLST_Project/Data/iTrialSpace/inserted_masks/mode1_controlled_prevalence/audit.json \
    --config tools/integration_config.yaml \
    --outdir /scratch/railabs/ft42/VLST_Project/Data/iTrialSpace/generated_cts/mode1_test \
    --case-ids 0 \
    -v
```

### 7.2 SLURM — All 9 Modes (sequential per mode)

```bash
cd /home/ft42/NoMAISI/slurm_nodmaisi
mkdir -p logs

# Dry run all modes
./submit_all_nodmaisi.sh --dry-run

# Submit all modes
./submit_all_nodmaisi.sh

# Submit selected modes
./submit_all_nodmaisi.sh 1 8
```

### 7.3 SLURM — Array Job (one case per GPU task)

```bash
cd /home/ft42/NoMAISI/slurm_nodmaisi

# Step 1: Generate case list from audit
AUDIT=/scratch/railabs/ft42/VLST_Project/Data/iTrialSpace/inserted_masks/mode1_controlled_prevalence/audit.json
python3 -c "
import json
audit = json.load(open('${AUDIT}'))
cases = sorted(set(str(r['case_id']) for r in audit['records'] if r['status']=='success'))
open('/tmp/mode1_controlled_prevalence_cases.txt','w').write('\n'.join(cases))
print(f'{len(cases)} cases')
"

# Step 2: Submit array job
N_CASES=$(wc -l < /tmp/mode1_controlled_prevalence_cases.txt)
sbatch --array=0-$((N_CASES - 1))%10 mode1_nodmaisi_array.sub
#                                 ^^ max 10 concurrent GPU tasks
```

### 7.4 From Mask Directory (no audit)

```bash
python tools/run_itrialspace_to_ct.py \
    --mask-root /scratch/.../inserted_masks/mode1_controlled_prevalence \
    --config tools/integration_config.yaml \
    --outdir /scratch/.../generated_cts/mode1 \
    -v
```

---

## 8. End-to-End Validation Checklist

### Pre-flight
- [ ] NodMAISI model checkpoints exist at configured paths
- [ ] `monai-auto3dseg` conda environment is activated
- [ ] `MONAI_DATA_DIRECTORY` is set
- [ ] iTrialSpace mask insertion has completed (audit.json exists)
- [ ] GPU is available (`nvidia-smi` shows free VRAM)

### Per-case Validation
- [ ] `input_mask.nii.gz` is RAS+ oriented, uint8
- [ ] Mask dimensions match a valid NodMAISI size
- [ ] `dataset.json` is well-formed
- [ ] NodMAISI inference completes without error
- [ ] `synthetic_ct.nii.gz` exists and is non-empty
- [ ] Synthetic CT shape matches input mask shape
- [ ] CT intensity range is within [-1000, 1000] HU
- [ ] No NaN values in synthetic CT
- [ ] Mask contains label 23 (nodule voxels)
- [ ] QC PNGs show anatomically plausible CT with visible nodule
- [ ] `nodmaisi_audit.json` has all expected fields

### Post-run
- [ ] `pipeline_summary.json` reports n_success > 0
- [ ] Failed cases have clear error messages
- [ ] QC PNGs can be visually inspected for artifacts
- [ ] Synthetic CT can be loaded in ITK-SNAP or 3D Slicer

---

## 9. Key Design Decisions

1. **Subprocess invocation** — NodMAISI is called via `subprocess.run()` rather than Python import, to avoid namespace pollution, ensure clean GPU state between cases, and match the existing SLURM workflow.

2. **Per-case dataset JSON** — Each case gets its own temporary `dataset.json` and `_tmp_env.json` to avoid modifying any shared NodMAISI config files.

3. **Nearest-neighbor resize** — When mask dimensions don't match NodMAISI's valid sizes, we resize using order=0 (nearest) to preserve integer label values. Spacing is adjusted proportionally.

4. **No changes to NodMAISI core** — All integration logic lives in `tools/` and `slurm_nodmaisi/`. The existing NodMAISI scripts and configs are untouched.

5. **Label 23 alignment** — iTrialSpace's default nodule label (23) matches NodMAISI's lung tumor label (23). No label remapping is needed. This is the critical compatibility point.

6. **Defensive processing** — Cases that fail are logged and skipped; processing continues. The summary report lists all failures.
