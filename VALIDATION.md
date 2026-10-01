# Published bundle validation

Validated on **2026-10-01** with a fresh clone on a GitHub-hosted Windows runner.

[Successful Windows restoration and offline inference run](https://github.com/lben/laya-model-bundle/actions/runs/36895995095)

| Checkpoint | Result | Inference examples | Load time (seconds) |
|---|---|---|---|
| english | PASS | 1 | 6.71 |
| multilingual | PASS | 2 | 6.94 |
| typed-decisions | PASS | 1 | 5.37 |

- All 82 compressed chunks were checked against their SHA-256 hashes.
- All 38 restored files matched the original snapshot SHA-256 hashes and lengths.
- Every checkpoint loaded from its restored local directory with socket connections blocked and Hugging Face offline mode enabled.
- Every example returned valid choice, ordinal score, and boolean probability outputs; multilingual inference also included Spanish input.
- Restored model files were verified again after inference.
- Nine restoration/corruption tests passed on Windows and Linux.
- Runtime: Python 3.12.10, PyTorch 2.8.0+cpu, Laya 0.3.22, CPU.

The workflow's `windows-inference-reports` artifact contains the full JSON outputs and installed runtime versions. These are functional smoke tests, not an accuracy benchmark. Timings describe the CI runner only.

The tested model/script commit was `7e7f89a`; subsequent documentation changes do not alter weights or test code. Upstream model revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
