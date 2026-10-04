# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ShinkaEvolve (`shinka`, PyPI `shinka-evolve`) evolves programs with LLMs as mutation operators inside an evolutionary loop: a population of programs is kept in a database (islands + archive), LLMs propose diffs/rewrites of parent programs, candidates are evaluated (locally or on Slurm), and scored results feed back into sampling. Python >= 3.10, managed with `uv`.

## Commands

```bash
uv sync --dev                     # install with dev deps

# CI checks (also run by .githooks/pre-push). Note ruff/mypy only cover tests/
uv run ruff check tests --exclude tests/file.py
uv run mypy --follow-imports=skip --ignore-missing-imports tests/test_*.py tests/conftest.py
uv run --with pytest-cov pytest -q -m "not requires_secrets and not models_dev_live" --cov=shinka

uv run pytest tests/test_llm_kwargs.py -q            # single file
uv run pytest tests/test_llm_kwargs.py::test_name -q # single test
uv run pytest -q -m requires_secrets                 # needs provider credentials

uv run --group docs mkdocs build --strict            # docs (mkdocs.yml, docs/)
```

Pytest markers: `integration`, `requires_secrets`, `models_dev_live` (`--strict-markers` is on).

Pricing CSVs (`shinka/llm/providers/pricing.csv`, `shinka/embed/providers/*.csv`) are generated from models.dev — never hand-edit; regenerate with `uv run python -m shinka.tools.pricing.generate_csvs` (`--check` to verify). Manual entries/aliases go in `shinka/tools/pricing/models_dev_overlay.json`.

CLI entry points (pyproject `[project.scripts]`): `shinka_run`, `shinka_launch` (Hydra-based), `shinka_models`, `shinka_visualize` (WebUI).

## Architecture

- `shinka/core/` — orchestration. `ShinkaEvolveRunner` (`async_runner.py`) is the single unified runner (it replaced the old `EvolutionRunner`/`AsyncEvolutionRunner`); configured by `EvolutionConfig` (`config.py`). Supporting pieces: `sampler.py` (prompt construction), `summarizer.py` (meta summaries), `novelty_judge.py` (+async variants), `prompt_evolver.py` (evolves system prompts), `wrap_eval.py` (`run_shinka_eval`, used by task `evaluate.py` files), `pipeline_timing.py`/`runtime_slots.py` (proposal-pipeline oversubscription and concurrency).
- `shinka/database/` — program archive (`dbase.py`, `async_dbase.py`), island model (`islands.py`, `island_sampler.py`), parent and inspiration selection, complexity metrics.
- `shinka/edit/` — applying LLM output to code: diff-style patches (`apply_diff.py`), full rewrites (`apply_full.py`), `marker_validation.py` for evolve-block markers; language-specific behavior is exercised in `tests/test_edit_*.py`.
- `shinka/llm/` — LLM client, query routing, kwargs, bandit-style model prioritization, and `providers/` (openai, anthropic, gemini, deepseek, local_openai, `headless` CLI-backed models such as `headless/codex@...`). `model_resolver.py` + `pricing.py` map model strings to provider and cost; accounting is per-provider (see `tests/test_*_accounting.py`). `shinka/embed/` is the analogous embedding layer.
- `shinka/launch/` — job execution backends (`local.py`, `slurm.py`, `scheduler.py`).
- `shinka/configs/` — Hydra config groups (`task`, `evolution`, `database`, `cluster`, `variant`) composed by `config.yaml`; `launch_hydra.py`/`eval_hydra.py` consume them.
- `shinka/webui/`, `shinka/plots/` — visualization; `shinka/cli/` — CLI implementations.
- `examples/<task>/` — each task has `initial.py` (seed program with evolve markers), `evaluate.py` (verifier), and `run_evo.py`. `skills/` holds agent skills (shinka-setup/convert/run/inspect) distributed via `npx skills`.

Startup conditionally refreshes the models.dev catalog (cached in the user cache, falling back to packaged CSVs); tests in `tests/test_models_dev_live.py` hit the network and are excluded from default CI.

## Conventions

- Add tests under `tests/` for bug fixes (see CONTRIBUTING.md); credit outside contributors in CHANGELOG.md/docs as recent commits do.
- Secrets load from `.env` via python-dotenv; do not commit it.
