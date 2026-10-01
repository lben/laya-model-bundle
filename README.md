# Laya System 1 — restorable model bundle

This repository contains the **actual official Hugging Face model files**, losslessly gzip-compressed into chunks of at most **45 MiB**. A normal Git clone retrieves all chunks; no Git LFS account, Hugging Face account, model download, or token is needed to restore them.

Included checkpoints from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya):

| Checkpoint | Restored directory | Purpose |
|---|---|---|
| English | `models/laya` | Original English System 1 model |
| Multilingual | `models/laya-multilingual` | Model for multilingual inputs |
| Typed decisions | `models/laya-typed-decisions` | Checkpoint fine-tuned for typed decision workflows |

`manifest.json` pins the upstream revision and records original file lengths, SHA-256 hashes, chunk order, and compressed chunk hashes. `source-lock.json` records upstream Git/LFS object identifiers. Model weights, tokenizers, encoder configs, decision configs, and all other snapshot files are preserved. The two bundled subfolders become independent local model directories. Compression does **not** quantize or change weights.

## Windows: clone, restore, and test

Install **Git** and **Python 3.12, 64-bit**, including the Python `py` launcher. Allow roughly **10 GB of free disk space** for the clone, restored files, dependencies, and installation overhead. Use **8 GB RAM or more**; inference tests load one model at a time. CPU inference works without a GPU.

In PowerShell:

```powershell
git clone --depth 1 https://github.com/lben/laya-model-bundle.git
cd laya-model-bundle
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -Install
```

This creates `.venv`, installs the pinned CPU runtime, restores all three models, verifies every file, and runs real **offline** inference for `choice`, `score`, and `noul`. The multilingual model is also tested on Spanish text. It prints `PASS` and writes JSON results to `reports/`. Package installation requires internet access; restoration and inference do not.

After the first installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1
```

For just the English model:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -Install -Model english
```

For NVIDIA CUDA instead, use `-Install -Device cuda`. This installs the PyTorch CUDA 12.8 wheel; a compatible NVIDIA driver is required. CPU is the configuration validated by the repository's Windows CI.

## Restore without installing any ML packages

Only the Python standard library is needed:

```powershell
py -3.12 restore.py                         # all models
py -3.12 restore.py --model english         # one model
py -3.12 restore.py --output D:\LayaModels  # another destination
py -3.12 restore.py --check-chunks          # verify compressed files only
py -3.12 restore.py --verify-only           # verify restored original files
```

Restoration streams across chunk boundaries, verifies hashes, and publishes each completed model directory atomically. It does not create a temporary full compressed archive. Each checkpoint resumes independently: rerunning skips an existing directory only if all its files match. Missing/corrupt chunks or changed restored files produce an error and a nonzero exit code. Changed model directories are preserved; rename them before restoring a fresh copy. Keep the chunks if you want to restore again.

## Use the restored model

Use the **Laya SDK**, which understands the decision head and typed questions:

```python
from pathlib import Path
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'
import laya

agent = laya.load(str(Path('models/laya').resolve()), device='cpu', compile=False)
result = agent.predict(
    'I was billed twice. Please refund the duplicate charge.',
    {'department': {
        'type': 'choice',
        'instructions': 'Which department should handle this?',
        'criteria': {'billing': 'invoices, payments, refunds',
                     'technical': 'bugs, outages, system errors'}
    }}
)
print(result)
```

Replace the path with `models/laya-multilingual` or `models/laya-typed-decisions` to use another checkpoint. The test checks that the model loads and produces finite, correctly shaped answers and probability distributions. It is a **smoke test**, not an accuracy benchmark. Upstream documents task accuracy and calibration limits in its model card; validate decisions on your own data.

## Validation and maintenance

- `python -m unittest discover -s tests -v` tests stream reconstruction, empty files, many chunk boundaries, corruption, missing chunks, original checksum failures, decompression size limits, safe paths, and preservation of changed files. CI runs these on Windows and Linux.
- The manually triggered **Windows restore and offline inference** workflow checks the published real weights, all restored SHA-256 hashes, and actual offline CPU inference on each checkpoint. Reports are saved as a workflow artifact.
- Maintainers can package a clean new bundle with `python scripts/pack_models.py --model english`, then `multilingual`, then `typed-decisions`. It streams the pinned upstream snapshot and validates Git blob/LFS digests before accepting each file. `--git-index` saves local disk space by storing chunks in Git's index and deleting working copies; it is intended for publishing from a staging checkout.
- Each chunk is below GitHub's [100 MB enforced per-object limit](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits). Upload model commits separately to stay below the 2 GB push limit. Avoid repeatedly committing replacement weight bundles, since binary history increases clone size.

## License and provenance

The original models and accompanying upstream files are published under Apache 2.0 by **Convai Innovations / Nandakishor M**. See [the upstream model card](https://huggingface.co/convaiinnovations/laya) and the preserved card at `models/laya/README.md` after restoration. Bundle scripts are also provided under Apache 2.0. `LICENSE` contains the full license; `NOTICE` retains attribution. The original model files are unmodified.
