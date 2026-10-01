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

The tested model/script commit was `7e7f89a`. The installer has since been simplified to install the same pinned runtime through your configured pip package index, without overriding it with a PyTorch download URL. Model weights, restoration, and inference checks are unchanged. Upstream model revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

## Offline interactive launcher validation

[Successful Windows local CLI and interactive-mode run](https://github.com/lben/laya-model-bundle/actions/runs/36905023238) on **2026-10-01**, commit `58ea5fa`.

- `laya-local.cmd` loaded restored local checkpoints with Hugging Face offline mode enabled and socket connections blocked.
- All five presets passed: triage, email, guard, moderation, and router.
- English, multilingual, and typed-decisions checkpoints all worked through the local launcher.
- An interactive session answered two separate prompts before `quit`, reusing its loaded model.
- Every restored file still matched its original hash after the launcher tests.
- All 17 unit tests passed on Windows and Linux, including rejection of missing local checkpoints without any Hub fallback.
- Installation used ordinary pip with its configured package source; the installer supplied no custom index URL.

The workflow artifact includes `local-*.json`, `local-interactive.txt`, the baseline inference reports, and runtime versions.
