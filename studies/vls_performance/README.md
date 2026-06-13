# VLS performance study (private layer)

The paper/study-specific analysis for the VLM benchmark. It is the modern,
non-drifting replacement for the old `paper_analysis/` folder: it holds **only**
study choices (which split, which figures, narrative/captions) and calls the
**public, reusable engine** for all logic:

```
itrialspace.evaluation.vlm_eval.eval_analysis
```

This folder is **gitignored** — it is *not* part of the public iTrialSpace release.
The engine it depends on *is* public, so anyone can reproduce the analysis; only
the manuscript-specific framing stays here.

## Why this split

The old paper analysis mixed engine + narrative in one external folder, so numbers
drifted from the data (e.g. 42,382 vs 42,716 in sibling files). Keeping the engine
in the package — tested, versioned, run in CI — means tables/figures are always
recomputed from the result CSVs, never hand-edited.

## Run

```bash
export ITRIALSPACE_DATA_DIR=/path/to/iTrialSpace
bash studies/vls_performance/run.sh                 # -> studies/vls_performance/output/
# options:
SPLIT=release_v1 NBOOT=2000 bash studies/vls_performance/run.sh out_dir/
```

Outputs (`output/`): `report.md`, `tables/*.csv`, `figures/*.{png,pdf}`.

## Engine reference

```bash
python -m itrialspace.evaluation.vlm_eval.eval_analysis \
    --results $ITRIALSPACE_DATA_DIR/vlm_dataset \
    --split   release_v1_full \
    --out     output/ \
    --n-boot  1000
```

- `--results` any results root (full dataset **or** a demo output dir)
- `--split` a split name under `vlm_dataset/splits/` or a file path; omit to use all cases
- produces accuracy / Δ-vs-plain / per-mode / per-dataset / per-size / per-lobe
  tables, confusion matrices, per-class accuracy, bootstrap CIs, and McNemar
  significance — see `docs/vlm_eval.md`.

For manuscript figures, add small scripts here that import
`itrialspace.evaluation.vlm_eval.eval_analysis.{tables,plots,stats}` and compose
exactly the panels/captions the paper needs.
