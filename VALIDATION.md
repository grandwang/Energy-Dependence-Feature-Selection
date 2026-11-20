# Validation — 2026-11-20

Tested on Windows x64, Python 3.9.13, with the versions in `requirements.txt`.

- Three inherited synthetic cases: selection order, independently fitted budget
  prefixes, relevance, shrinkage, correlation columns, objective, and gains pass.
- Three real datasets / nine budget records: all selected feature prefixes match
  the frozen historical references exactly; all 36 3-NN metric values pass the
  `1e-14` absolute tolerance.
- Both the original processed files and freshly obtained upstream files pass
  the benchmark. Upstream and local arrays have identical values and ordering.
- Corrupted data bytes and overlapping train/test indices are rejected.
- The scientific `selector.py` differs only in 11 selected internal variable
  names. Reversing those names produces an identical Python AST to the supplied
  frozen source. Public interfaces, literals, and arithmetic are unchanged.

The upstream files were downloaded successfully using PowerShell. A direct
Python urllib download attempt in the validation environment stalled and was
interrupted; end-to-end automatic Python downloading is **not certified** in
this environment. Manual download plus hash-checked execution is validated.
The existing-file verification path in `download_data.py` is also tested.

The environment reused an existing installation of the pinned dependencies;
a new pip installation, Linux/macOS execution, and full-paper reproduction
were not tested. No unavailable historical prediction/process reference has
been fabricated.
