# Changelog

## 0.5.1

The capability inventory includes install, invoke, remember, and skills. The Windows installer tries a CUDA 13 wheel of `llama-cpp-python` before the older CUDA 12.4 wheel and the CPU wheel.

## 0.5.0

An unfamiliar objective names the capability it lacks, tries that capability, and stores the procedure when the expected result appears. A later objective that names the saved skill reuses it instead of repeating the trial.

## 0.4.0

Modules can be registered from a project folder, installed into the project virtual environment, and called without changing the core. The registration and any saved skill remain after a restart.

## 0.3.0

An objective can search the web and GitHub, read a PyPI package, fetch a page, and record dependencies and install lines found in documentation. Downloads and source metadata stay in SQLite.

## 0.2.0

Background objectives. A request is planned into steps, run after the HTTP call returns, and resumed after a restart. GGUF files in subfolders of `models/` are discovered.

## 0.1.0

YAML configuration, SQLite conversation state, JSON logs, and a FastAPI process that loads a GGUF dropped into `models/`.
