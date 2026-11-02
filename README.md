# Genesis

Local HTTP service for [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) language models.

Genesis loads a model from a folder on your machine, runs inference in-process, and stores every conversation in SQLite. The server binds to loopback. No cloud model API is required.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)

## Features

- Point the server at a directory of `.gguf` files. One file loads on its own.
- When several files are present, Genesis lists them and waits for a choice. That choice is kept across restarts.
- Chat completions over HTTP, including a server-sent event stream.
- Per-conversation history stored in SQLite.
- Context length, sampling, GPU layers, and paths live in YAML.
- Structured JSON logs for each request.

A model file is not included.

## Requirements

- Windows 11
- Python 3.11 or newer
- A `.gguf` model

GPU offload needs a CUDA build of `llama-cpp-python` whose native library actually loads. The installer tries a CUDA wheel inside the project virtual environment, then a CPU wheel. A CPU build ignores GPU layers.

## Quick start

From the repository root in PowerShell:

```powershell
.\scripts\install.ps1
.\scripts\run.ps1
```

`install.ps1` creates `.venv` when it is missing and installs every Python package there. `run.ps1` starts the API with `.venv\Scripts\python.exe`. The process exits if it is started with any other interpreter.

The server listens on [http://127.0.0.1:8765](http://127.0.0.1:8765).

Check that it is up:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

```json
{
  "status": "ok",
  "venv": "C:\\git\\AI_Worker\\.venv",
  "phase": "0"
}
```

## Add a model

Put one `.gguf` file directly in `models/` (not in a subfolder). Restart the server, or rescan while it is running:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/model/reload
```

See what was found:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/model
```

With a single file, `loaded` is `true` and `filename` is that file. With several files, `loaded` stays `false` until you select one:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/model/select `
  -ContentType application/json `
  -Body '{"filename":"your-model.gguf"}'
```

The selection is stored in `storage/genesis.db`. To pin a file from configuration instead, set `model.filename` in `config/genesis.yaml`. While that value is set, `POST /model/select` will not replace it.

`model.n_gpu_layers` defaults to `0`, so inference stays on the CPU. Set it to `-1` to offload every layer to an NVIDIA GPU. That only has an effect when the installed `llama-cpp-python` build was compiled with CUDA and its DLL loads.

## Chat

Start a conversation:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/chat `
  -ContentType application/json `
  -Body '{"message":"Hello"}'
```

```json
{
  "conversation_id": "b6f1c2e0-4d3a-4f0e-9c2a-1a2b3c4d5e6f",
  "message": "Hello.",
  "model": "your-model.gguf",
  "prompt_tokens": 12,
  "completion_tokens": 2,
  "duration_ms": 840.2
}
```

Continue it by sending `conversation_id` with the next message. Read it back after a restart:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/conversations
Invoke-RestMethod http://127.0.0.1:8765/conversations/<conversation_id>
```

`POST /chat/stream` returns `text/event-stream`. Each event is a `data: ` line of JSON.

| `type` | Meaning |
| --- | --- |
| `delta` | One piece of assistant text in `delta` |
| `done` | The finished `message`, model name, and `duration_ms` |

Without a loaded model, chat returns `409` and the model status explains what is in the folder.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process status and virtualenv path |
| `GET` | `/model` | Loaded file, discovered files, and load detail |
| `POST` | `/model/reload` | Rescan the models directory |
| `POST` | `/model/select` | Body: `{"filename":"model.gguf"}` |
| `POST` | `/chat` | Send a message. Optional `conversation_id`, `temperature`, `max_tokens` |
| `POST` | `/chat/stream` | Same body as `/chat`, streamed as server-sent events |
| `GET` | `/conversations` | Conversations, newest first |
| `GET` | `/conversations/{id}` | One conversation and its messages |

Interactive request docs are served at [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs) while the server is running.

## Configuration

`config/genesis.yaml` controls the running service. Editing it does not require a code change. Restart the server after a change.

| Key | Default | Role |
| --- | --- | --- |
| `server.host` | `127.0.0.1` | Bind address |
| `server.port` | `8765` | Bind port |
| `model.directory` | `models` | Folder of `.gguf` files, relative to the repo or absolute |
| `model.filename` | unset | Pin one filename. Leave unset to auto-load the only file, or to use the API selection |
| `model.context_length` | `4096` | Context window |
| `model.n_gpu_layers` | `0` | `0` is CPU. `-1` offloads every layer when the CUDA build loads |
| `model.temperature` | `0.7` | Sampling temperature |
| `model.top_p` | `0.95` | Nucleus sampling |
| `model.max_tokens` | `1024` | Reply limit |
| `database.path` | `storage/genesis.db` | SQLite file |
| `logging.level` | `INFO` | Python log level |
| `logging.directory` | `storage/logs` | JSON logs are written to `genesis.jsonl` here |
| `identity.name` | `Genesis` | Assistant name |
| `identity.system_prompt` | see the yaml | System message sent with each conversation |

Environment overrides:

| Variable | Overrides |
| --- | --- |
| `GENESIS_CONFIG` | Path to the YAML file |
| `GENESIS_MODEL_DIR` | `model.directory` |
| `GENESIS_MODEL_FILENAME` | `model.filename` |
| `GENESIS_ROOT` | Repository root used to resolve relative paths |

## Development

Use the project virtual environment for every install, test, and run. Do not install packages into the system Python.

```powershell
.\scripts\install.ps1
.\scripts\test.ps1
.\scripts\run.ps1
```

`scripts/test.ps1` runs `pytest` with `.venv\Scripts\python.exe`. Direct `pytest` or `python` from outside `.venv` is rejected.

Logs and the database are created under `storage/` on first launch. That directory is gitignored.

## Project layout

```text
config/genesis.yaml          Runtime configuration
genesis/                     Application package
  api/                       HTTP routes
  core/configuration/        YAML settings
  core/model_interface/      GGUF discovery and inference
  core/conversation.py       Conversation history
  monitoring/                JSON logging
  storage/                   SQLite access
models/                      Drop .gguf files here
scripts/                     install.ps1, run.ps1, test.ps1
tests/                       Pytest suite
```
