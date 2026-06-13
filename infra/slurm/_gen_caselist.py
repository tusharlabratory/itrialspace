#!/usr/bin/env python3
"""Write a synthesis case list (mask-filename stems) from an insertion audit.json.

The synthesis runner matches --case-ids against the mask-filename stem (NOT the
audit's integer case_id), so the case list must contain those stems.

Usage: python _gen_caselist.py <audit.json> <out_caselist.txt>   # prints N cases
"""
import json
import os
import sys


def stem(path: str) -> str:
    name = os.path.basename(path)
    for suffix in ("_mask.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> None:
    audit, out = sys.argv[1], sys.argv[2]
    data = json.load(open(audit))
    recs = data.get("records", data.get("cases", []))
    cases = sorted(
        {stem(r["output_combined_path"]) for r in recs
         if r.get("status") == "success" and r.get("output_combined_path")}
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(cases) + "\n")
    print(len(cases))


if __name__ == "__main__":
    main()
