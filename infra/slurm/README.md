# SLURM job scripts

HPC submission scripts for every stage of the iTrialSpace pipeline. They are
**cluster-portable**: paths, the conda environment, and partitions come from
environment variables (with sane defaults) rather than being hardcoded.

## How portability works

Every script sources [`env.sh`](env.sh) near the top:

```bash
source "$(dirname "$0")/../env.sh"   # sets PROJ_DIR, ITRIALSPACE_DATA_DIR, CONDA_ENV, ...
itrialspace_activate_conda            # activates the conda env portably
```

`env.sh` resolves (overridable via your shell or a repo-root `.env`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `PROJ_DIR` | auto (repo root) | iTrialSpace checkout |
| `ITRIALSPACE_DATA_DIR` | `~/.itrialspace/data` | unified data layout |
| `ITRIALSPACE_OUTPUT_DIR` | `=DATA_DIR` | generated artifacts |
| `ITRIALSPACE_CONDA_ENV` | `itrialspace` | conda env name |
| `CONDA_PROFILE` | auto-detected | path to `conda.sh` |
| `SLURM_PARTITION_GPU` / `SLURM_PARTITION_CPU` | `gpu` / `cpu` | partitions |

## Adapting to a new cluster

1. Set the variables above in your `.env` (copy `.env.example`) or shell profile.
2. **Edit the `#SBATCH` directives** in the scripts you use — partition names
   (`--partition`), account (`--account`), GPU type, walltime, and memory are
   cluster-specific and cannot be set from environment variables (SLURM parses
   `#SBATCH` lines before the shell runs). Search-and-replace the partition names
   for your site.
3. Submit from the repo root, e.g. `sbatch infra/slurm/nodulemap/nodulemap.sub`.

## Layout

| Directory | Stage |
|-----------|-------|
| `trials/` | Generate cohort manifests (13 modes) |
| `mask_inserter/` | Insert nodule masks into host CTs |
| `synthesis/` | NodMAISI CT synthesis (needs model weights — see docs) |
| `nodulemap/` · `retriever/` | Launch interactive web tools |
| `vlm_eval/` | Vision-language model evaluation |

> Logs are written to `logs/` (gitignored). Create it first or let the scripts `mkdir -p logs`.
