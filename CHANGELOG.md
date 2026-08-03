# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.1] — 2026-06-26

### Changed

- `README.md` — added Real-World Applications section (8 enterprise use cases across government contracting, procurement, legal, financial, customer support, compliance, HR, and records management)
- `docs/MODEL_CARD.md` — strengthened Project Objective section; reframed as an end-to-end supervised ML workflow emphasizing explainability, reproducibility, maintainability, and production-oriented engineering

---

## [1.0.0] — 2026-06-25

### Added

- `.github/workflows/tests.yml` — CI pipeline; runs `pytest tests/ -v` on push and pull_request (ubuntu-latest, Python 3.11)
- `LICENSE` — MIT License (2026 Richie Garafola)
- `CHANGELOG.md` — this file; Keep-a-Changelog format
- `docs/ARCHITECTURE.md` — system architecture, module map, data-flow diagram, TrainResult dataclass reference, coefficient extraction logic, CV methodology, design decisions
- `docs/TESTING.md` — test strategy, corpus design, per-class test inventory (52 tests across 8 classes), CI integration
- `docs/DATA_DICTIONARY.md` — complete schema reference for TrainResult (13 fields), all function output DataFrames, summary_stats dict keys, sample CSV columns
- `docs/MODEL_CARD.md` — model card covering project objective, classification task, algorithms, feature engineering, training workflow, evaluation methodology, performance characteristics, limitations, and intended use cases
- `.streamlit/config.toml` — light theme, telemetry disabled
- `screenshots/.gitkeep` — placeholder for post-modernization screenshot session

### Changed

- `app/main.py` — removed `Day 18 — Document Classifier` series language from module docstring; replaced with neutral project description
- `src/__init__.py` — populated with 11 public exports from classifier, metrics, and vectorizer submodules (previously empty)
- `README.md` — added CI and Python version badges; replaced `<your-username>` clone URL placeholder with `RichieGarafola`; added Skills Demonstrated section; added Machine Learning Workflow section
