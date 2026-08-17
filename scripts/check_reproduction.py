#!/usr/bin/env python
"""Compare two results_macros.tex files and fail on any shared-macro mismatch.

Guards the reproducibility contract: a pipeline re-run may ADD macros but must
reproduce every previously published value exactly (at displayed precision).

Usage:
    python scripts/check_reproduction.py <reference_macros.tex> <regenerated_macros.tex>

Exit code 0 if all shared macros match, 1 otherwise.
"""
import re
import sys

MACRO_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}")


def parse(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = MACRO_RE.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main(ref_path, new_path):
    ref, new = parse(ref_path), parse(new_path)
    if not ref:
        print(f"ERROR: no macros parsed from {ref_path}")
        return 1
    missing = sorted(set(ref) - set(new))
    mismatched = sorted(k for k in set(ref) & set(new) if ref[k] != new[k])
    added = sorted(set(new) - set(ref))
    for k in mismatched:
        print(f"MISMATCH  \\{k}: reference={ref[k]!r}  regenerated={new[k]!r}")
    for k in missing:
        print(f"MISSING   \\{k}: present in reference, absent in regenerated file")
    if added:
        print(f"added (ok): {', '.join(added)}")
    if mismatched or missing:
        print(f"FAILED: {len(mismatched)} mismatched, {len(missing)} missing "
              f"of {len(ref)} reference macros")
        return 1
    print(f"OK: all {len(ref)} reference macros reproduced exactly; "
          f"{len(added)} new macros added")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
