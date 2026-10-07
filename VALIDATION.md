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
# Large option lists: token-budget validation

On **2026-10-07**, the restored English checkpoint was tested with real offline CPU inference on macOS ARM64 using the repository's pinned runtime (`laya==0.3.22`, `torch==2.8.0`, `transformers==4.57.6`). This test used 379 synthetic category labels/descriptions and a short support message, passed through files to `laya_local.py`.

- Default settings reproduced `question 'bucket': only 126 of its 379 option markers fit in max_len=512 with head_max_len=192 spent on the question`.
- `--max-len 4096 --head-max-len 3072` succeeded and returned all 379 category probabilities. Every value was finite and between 0 and 1; their sum was 1.0001 (SDK rounding).
- SDK usage reported 3064 input tokens, 16 message tokens, 0 dropped message tokens, and `truncated: false`.
- The CLI's network guard was active during model loading and inference. The restored model was verified against the manifest before loading.
- This checks input fitting and output shape, not accuracy on the user's categories or confidence calibration. The SDK warned that the English checkpoint's `choice:11+` temperature was clamped from approximately 0.1006 to 0.5 and that affected confidence is uncalibrated.

The synthetic list was generated as `category_000=customer support requests about topic 000` through `category_378=customer support requests about topic 378`. The message was `Please help me with billing and customer support topic 123.` The 57 unit/API tests also passed locally, covering budget forwarding for CLI options, plain-text requests, presets, interactive calls, Python classification, and HTTP requests.
