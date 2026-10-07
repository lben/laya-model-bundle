# Laya System 1 — restorable model bundle

This repository contains the **actual official Hugging Face model files**, losslessly gzip-compressed into chunks of at most **45 MiB**. A normal Git clone retrieves all chunks; no Git LFS account, Hugging Face account, model download, or token is needed to restore them.

Included checkpoints from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya):

| Checkpoint | Restored directory | Purpose |
|---|---|---|
| English | `models/laya` | Original English System 1 model |
| Multilingual | `models/laya-multilingual` | Model for multilingual inputs |
| Typed decisions | `models/laya-typed-decisions` | Checkpoint fine-tuned for typed decision workflows |

The complete bundle contains **82 chunks (2.01 GiB compressed)** and restores to **2.21 GiB** of original files. The pinned source revision is `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

`manifest.json` pins the upstream revision and records original file lengths, SHA-256 hashes, chunk order, and compressed chunk hashes. `source-lock.json` records upstream Git/LFS object identifiers. Model weights, tokenizers, encoder configs, decision configs, and all other snapshot files are preserved. The two bundled subfolders become independent local model directories. Compression does **not** quantize or change weights.

## Windows: clone, restore, and test

Install **Git** and **Python 3.12, 64-bit**, including the Python `py` launcher. Allow roughly **10 GB of free disk space** for the clone, restored files, dependencies, and installation overhead. Use **8 GB RAM or more**; inference tests load one model at a time. CPU inference works without a GPU.

In PowerShell:

```powershell
git clone --depth 1 https://github.com/lben/laya-model-bundle.git
cd laya-model-bundle
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -Install
```

This creates `.venv`, installs the pinned runtime dependencies, restores all three models, verifies every file, and runs real **offline** inference for `choice`, `score`, and `noul`. Installation uses a normal `pip install -r requirements.txt`, including PyTorch, and respects your existing `pip.ini` and configured package index (such as company Artifactory). It does not override the package source or upgrade pip. The multilingual model is also tested on Spanish text. It prints `PASS` and writes JSON results to `reports/`. Installation needs access to your configured package registry; restoration and inference do not.

After the first installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1
```

For just the English model:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -Install -Model english
```

For NVIDIA CUDA instead, use `-Device cuda` with a CUDA-enabled PyTorch build and a compatible NVIDIA driver. The device option selects where inference runs; it does not change your pip package source. CPU is the configuration validated by the repository's Windows CI.

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

## Offline interactive mode and prompts

From the repository folder, start the local interactive launcher:

```powershell
.\laya-local.cmd --preset triage
```

Enter text at `laya>` and type `quit` or `exit` to finish. The model is loaded once and reused for every entry. Available presets are `triage`, `email`, `guard`, `moderation`, and `router`. These return typed decisions and probabilities about your text.

For a single prompt, JSON output, another checkpoint, or a custom restoration destination:

```powershell
.\laya-local.cmd "I was charged twice. Please refund me." --preset triage --json
.\laya-local.cmd --preset moderation
.\laya-local.cmd --preset triage --model multilingual
.\laya-local.cmd --preset triage --models-dir D:\LayaModels
```

This launcher uses your existing `.venv` and defaults to the restored English checkpoint at `models/laya` on CPU. Select `--model multilingual` or `--model typed-decisions` for the other local directories, or `--device cuda` for a CUDA-enabled runtime. It verifies the restored files, loads the model by its absolute local path, enables Hugging Face offline mode, and blocks socket connections during loading and inference. If a model is missing, it tells you to restore it locally. It does not download model files.

Use **`laya-local.cmd`** for this repository. The package's separate `.venv\Scripts\laya.exe` CLI defaults to Hub model IDs and does not automatically use these restored folders.

The cross-platform equivalent is `python laya_local.py --preset triage` using your environment's Python.

### Custom buckets with `--options`

Put the message first, then a comma-separated list of categories. No message flag or JSON input is needed:

```powershell
.\laya-local.cmd "I was charged twice. Please refund me." --options "billing,technical,sales,other"
.\laya-local.cmd "I was charged twice." --options "A=billing,B=technical support,C=sales,D=other" --json
```

The command prints the winning category and a probability for every option. Probabilities sum to approximately 1. Surround the list with quotes, especially for names containing spaces. Spaces around commas are ignored; provide at least two distinct, nonempty options. Optional `label=description` or `label:description` entries give short labels a meaning. Commas separate entries, so names/descriptions must not contain commas.

Omit the message for interactive mode with the same options on every entry:

```powershell
.\laya-local.cmd --options "billing,technical support,sales,other"
```

Use either `--options` or `--preset`. With `--options`, the message is passed directly to the classifier, including any instruction-like wording it contains. The existing local checkpoint, device, and JSON flags also work.

### Read the message and options from files

For large inputs, pass file paths instead of expanding file contents in PowerShell:

```powershell
.\laya-local.cmd "@.\message.txt" --options "@.\options.txt" --json
```

`message.txt` contains the complete message, including any line breaks. `options.txt` contains the usual comma-separated list, for example:

```text
billing,technical support,sales,other
```

You can also use explicit flags:

```powershell
.\laya-local.cmd --text-file .\message.txt --options-file .\options.txt --json
.\laya-local.cmd "Inline message" --options-file .\options.txt
.\laya-local.cmd "@.\message.txt" --options "billing,other"
.\laya-local.cmd --options-file .\options.txt  # interactive with fixed options
```

Quote the entire `"@path"` argument in PowerShell, and quote any path containing spaces. Relative paths are resolved from your current directory. Files can be UTF-8 (with or without a BOM) or UTF-16 with a BOM, as commonly produced by Windows PowerShell. The message file's whitespace and line breaks are preserved; option entries are trimmed as usual. Missing, empty, unreadable, or invalid files produce an error before loading the model. Use either positional text or `--text-file`, and select only one of `--preset`, `--options`, or `--options-file`. A literal inline value beginning with `@` can be escaped as `@@`.

Only the file paths travel through the shell, avoiding command-length limits. The model's context window still applies to the loaded text.

### Large option lists and token limits

If Laya reports `only ... option markers fit in max_len=512`, its model input is too short to hold the whole question. The file was read successfully; this is the inference token limit. The English checkpoint defaults to `max_len=512` for the entire input and `head_max_len=192` for formatting the question/options. These limits count tokenizer tokens, not characters or file bytes.

To keep a large list such as 379 options, start with:

```powershell
.\laya-local.cmd "@.\message.txt" --options "@.\options.txt" --max-len 4096 --head-max-len 3072 --json
```

`--max-len` raises the total input limit. `--head-max-len` gives more room to the question and option descriptions; increase the total limit with it so the message still has room. The SDK may shorten individual option descriptions to fit the head budget, even when every marker fits. This command keeps every option in one choice question; it does not shortlist or split/renormalize probabilities across batches. If your descriptions or message need more space, the bundled encoders support up to 8192 positions: try `--max-len 8192 --head-max-len 6144`. Actual message room depends on the rendered question length. Long messages can still be truncated by the SDK, and larger settings cost more memory and time. Validate accuracy with your own messages/options; fitting all options is not an accuracy guarantee.

These flags also work with presets and interactive mode. Omitting them preserves checkpoint defaults. Both values must be positive, and an explicitly supplied head budget must be below an explicitly supplied total limit. The controls are forwarded per call, without editing the restored checkpoint files.

The Python client accepts the same settings:

```python
result = model.classify(message, options, max_len=4096, head_max_len=3072)
```

For the local HTTP API, include `"max_len": 4096` and `"head_max_len": 3072` in the request JSON, or set server defaults with `python -m laya_api --model-path .\models\laya --max-len 4096 --head-max-len 3072`.

### Custom buckets in plain text

No JSON file is needed. At `laya>`, enter this supported sentence format:

```text
From the following options billing, technical, sales and other categorize the following message on each one: “I was charged twice. Please refund me.”
```

You can also give letter labels explicit meanings:

```text
From the following options A=billing | B=technical support | C=sales | D=other categorize the following message on each one: “I was charged twice. Please refund me.”
```

The launcher recognizes that sentence format and builds the structured question locally. It prints the winning option and the probability for **every** option. These are competing category probabilities, summing to approximately 1. No additional model or network service is used to parse the sentence.

Separate options with commas, semicolons, or `|`; single-word options can also be separated by spaces. Descriptions can use `=` or `:`, with delimiters between options that contain spaces. Straight quotes and curly quotes around the message are both accepted, and quotes can be omitted. Bare `A B C and D` is accepted, but letters alone do not explain what the categories mean to the model.

This format chooses custom buckets for that entry. Ordinary entries continue using the selected preset. It is a supported input format, rather than a general-purpose natural-language instruction parser.

## Use the restored model from Python

### Easy import from another project

After `git pull`, activate **your other project's Python environment** and install this local checkout:

```powershell
python -m pip install C:\Projects\laya-model-bundle
```

Change that path to your clone's directory. This uses your configured pip index (including company Artifactory), installs the pinned runtime dependencies, and does not create a virtual environment. If that environment already has every dependency from `requirements.txt`, you can add `--no-deps`. The installed package contains just the Python client/server code; the compressed chunks and restored weights stay in your clone. No `sys.path` edits, subprocess calls, or working-directory changes are needed.

In your project:

```python
from laya_client import LocalLaya

# Create this once at application startup, then reuse it for every request.
model = LocalLaya(r"C:\Projects\laya-model-bundle\models\laya")
result = model.classify(
    "I was charged twice.",
    ["billing", "technical", "sales", "other"],
)
print(result["choice"])
print(result["probabilities"])  # probability for every option
```

`options` also accepts `"billing,technical,sales,other"` or a dictionary such as `{"A": "invoices and refunds", "B": "technical support"}`. `classify` returns the SDK's bucket answer, including `choice`, `probabilities`, and confidence fields. These are competing category probabilities, summing to approximately 1. Use `model.predict(state, questions, **kwargs)` for the complete SDK interface and other typed questions.

The path must point at an already restored checkpoint folder. Other checkpoints are `models/laya-multilingual` and `models/laya-typed-decisions`. Pass `device="cuda"` for an available CUDA runtime. The client checks required local files, sets Hugging Face offline flags before importing the model runtime, loads an absolute local path, and preserves the tokenizer configuration. Restore/verify the trusted snapshot with `restore.py`; the importable client does not repeat full weight checksums on startup. Initialize it before importing other Hugging Face runtimes in your application. It leaves your application's HTTP/socket connections available. Calls on one client instance are serialized; create one per application process, rather than one per request.

### Local HTTP API for any language

The installed package also includes a small HTTP adapter with no extra web-framework dependencies. Start it in an environment with the runtime installed:

```powershell
python -m laya_api --model-path C:\Projects\laya-model-bundle\models\laya --port 8000
```

Or run it from the clone using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m laya_api --model-path .\models\laya
```

It loads the model once, then serves `POST http://127.0.0.1:8000/classify` with this JSON body:

```json
{"message": "I was charged twice.", "options": ["billing", "technical", "sales", "other"]}
```

The response is the same dictionary returned by `classify`, with `choice` and `probabilities`. `options` can also be a comma-separated string or a description dictionary. `GET /health` returns `{"ready": true}` after startup. Invalid input returns HTTP 400; request bodies are limited to 1 MiB. This adapter binds to localhost and has no authentication; it is intended for trusted applications on the same machine. For a remote deployment, put the Python client behind your application's authenticated API. Browser frontends should call through their backend; this adapter does not enable CORS.

For example, call it from another Python project using only the standard library:

```python
import json
from urllib.request import Request, urlopen

request = Request(
    "http://127.0.0.1:8000/classify",
    data=json.dumps({"message": "I was charged twice.", "options": "billing,technical,sales,other"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choice"], result["probabilities"])
```

### Direct upstream SDK

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

All three checkpoints passed restoration and real offline CPU inference on Windows on 2026-10-01. See [VALIDATION.md](VALIDATION.md) for the successful run, runtime versions, and reports.

- `python -m unittest discover -s tests -v` tests stream reconstruction, empty files, many chunk boundaries, corruption, missing chunks, original checksum failures, decompression size limits, safe paths, and preservation of changed files. It also tests local CLI preset selection, absolute checkpoint paths, network blocking, interactive model reuse, and rejection of missing/changed checkpoints. CI runs these on Windows and Linux.
- Client/API tests cover model reuse, all three options formats, missing checkpoint files, tokenizer preservation, and real localhost HTTP requests with input errors and body limits. They mock the ML runtime; real checkpoint inference is validated separately by the manual Windows workflow.
- The manually triggered **Windows restore and offline inference** workflow checks the published real weights, all restored SHA-256 hashes, and actual offline CPU inference on each checkpoint. Reports are saved as a workflow artifact.
- Maintainers can package a clean new bundle with `python scripts/pack_models.py --model english`, then `multilingual`, then `typed-decisions`. It streams the pinned upstream snapshot and validates Git blob/LFS digests before accepting each file. `--git-index` saves local disk space by storing chunks in Git's index and deleting working copies; it is intended for publishing from a staging checkout.
- Each chunk is below GitHub's [100 MB enforced per-object limit](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits). Upload model commits separately to stay below the 2 GB push limit. Avoid repeatedly committing replacement weight bundles, since binary history increases clone size.

## License and provenance

The original models and accompanying upstream files are published under Apache 2.0 by **Convai Innovations / Nandakishor M**. See [the upstream model card](https://huggingface.co/convaiinnovations/laya) and the preserved card at `models/laya/README.md` after restoration. Bundle scripts are also provided under Apache 2.0. `LICENSE` contains the full license; `NOTICE` retains attribution. The original model files are unmodified.
