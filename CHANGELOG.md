# Changelog

## 0.9.1

An objective continues when the learning reply is not JSON. That reply is asked for action names and saved skills, and a trial succeeds when the expected text appears in the model reply or in a tool observation. A step asks again when a reply looks like JSON but is not one action, asks the user for more information, or repeats an action it already ran. A module that is already in the project is reached with inspect, install, and invoke, and a web lookup for that module is turned back.

## 0.9.0

An objective can run one local program and read page text with browse. Both stay off the public internet for private addresses until `lab.enabled` is set, and `run` stays off until `execution.enabled` is set. A learning experiment is kept only when it names actions Genesis already has. One step can use eight actions.

## 0.8.1

Sending a chat message shows that message at once, then “Message sent. Waiting for a reply.” until the model answers.

## 0.8.0

A workstation at `/workstation` shows the dashboard, live objective monitor, objectives, source archive, capabilities, and the running configuration. The pages are HTML served by the same process and read the existing JSON API.

## 0.7.0

Tested source generations stay in an archive together. A parent is chosen by task success, then by elapsed time. An earlier generation can be read and used as the starting point for another candidate. The live source is not replaced.

## 0.6.0

A source change can be written into a separate candidate, tested there, and kept in an archive. The live files stay as they were, so the previous version remains recoverable.

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
