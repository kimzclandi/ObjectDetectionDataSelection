# Data Selection and Controlled Training for Object Detection

[简体中文](README.md) | **English**

![Project wordmark](.github/project-header.svg)

[![CI](https://github.com/kimzclandi/ObjectDetectionDataSelection/actions/workflows/ci.yml/badge.svg)](https://github.com/kimzclandi/ObjectDetectionDataSelection/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/kimzclandi/ObjectDetectionDataSelection?style=flat)](https://github.com/kimzclandi/ObjectDetectionDataSelection/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Which images should be added to detector training under a fixed additional-image budget? This BDD100K subset study implements random, uncertainty and diversity selection, plus targeted selection using prototypes derived from development-set misses. All arms share pretrained Faster R-CNN MobileNetV3 and **finetune only the ROI predictor head**. Current experiments do not show a stable advantage of targeted selection over random selection.

This is a personal research project constrained by local resources. Inputs are road images and existing annotations. Outputs include selected-image lists, training records, per-image predictions and grouped evaluations. Annotations are supplied to training after selection to simulate newly labeled data.

## Features

- Random, uncertainty and diversity selection under a fixed budget.
- ROI predictor-head training, failure slices and targeted-selection diagnostics.
- Locked dependencies, evidence verification and result viewers.

## Components and data flow

| Layer | Input → output | Constraints |
|---|---|---|
| Ingestion | Images/annotations → verified metadata and anomaly records | Decode, box and hash checks; original images prepared separately |
| Inference recovery | Images/fixed model → cached predictions and task state | SQLite commits allow repeated computation but prevent duplicate committed results |
| Selection | Candidate pool/development failures → fixed-size image lists | Shared budgets; existing labels simulate new annotation |
| Training/evaluation | Common starting point/lists → per-image predictions, AP and slices | Matched steps, three seeds, ROI predictor head only |

## Reading and verification

The [experiment guide](docs/EXPERIMENT_GUIDE.md) covers inputs/outputs, controls, metric denominators, code/evidence paths and the distinction between verifying saved records and rerunning experiments. Start with results and limitations below, then trace records. Read environment and output-protection instructions before running commands. Linked technical documents retain their original language.

## Project history (added 2026-09-20)

According to the maintainer, related early work began locally around July–August 2026 before consolidation and upload to GitHub. This approximate starting point does not date every current feature or experiment. Later implementations, experiments and maintenance retain their actual version and run dates.

## Current results

The latest comparison uses a new 120-image evaluation set and three random seeds. Every arm trains for 112 steps; targeted and random selection each add 12 images. Results are separate from the earlier 240-image pilot.

| Method | Test AP × 100, mean over three seeds |
|---|---:|
| Targeted selection | 8.595 |
| Random selection | 9.229 |
| Seed-only (repeat existing images; match steps) | 7.609 |

Targeted minus random is **−0.634 AP points**, with a prefix-group paired bootstrap interval of **[−1.631, +0.323]**. The interval crosses zero, so a selection gain cannot be claimed. Later development diagnostics found a 0.988 rank correlation between the full score and its low-score quality term, and a leave-group-out prototype AUC of 0.395. These are association diagnostics, not a causal explanation of the AP difference; they did not lead to additional training.

[Training report](docs/FAILURE_V2_REPORT.md) · [Development diagnostics](docs/DIAGNOSIS_V3_REPORT.md) · [Training records](reports/failure_v2/summary.json) · [Diagnostic records](reports/diagnosis_v3/result.json)

## Quick start

Install locked dependencies from the repository root and inspect saved results first. This path does not download images or train models.

### Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```bash
git clone https://github.com/kimzclandi/ObjectDetectionDataSelection.git
cd ObjectDetectionDataSelection
uv sync --locked --python 3.12 --extra dashboard --extra dev
```

### Usage

```bash
uv run python scripts/verify_artifacts.py
uv run streamlit run dashboard.py --server.address 127.0.0.1
```

The default view, **「失败驱动训练 · 新评测集」** (failure-driven training, new evaluation set), shows the table above. 「历史训练 / VLM 推理」 displays earlier experiments. Default port: 8501; use `--server.port 8512` to choose another. Open development diagnostics with `uv run streamlit run diagnosis_dashboard.py`.

![Saved failure-driven training results](assets/failure-v2-dashboard.png)

The screenshot comes from the saved training experiment, not later diagnostics or new training. Metrics and records are available without original images; image viewing and model reruns require separately prepared data and fixed weights.

[Running: replay, data and training](docs/RUNNING.md) · [Protocol and recovery](docs/FAILURE_V2_PROTOCOL.md) · [Data card/licensing](docs/DATA_CARD.md)

## Implementation and attribution

Repository work includes splitting/validation, candidate scoring, fixed-budget comparisons, predictor-head training, sliced evaluation and result browsing. Detector architecture and pretrained weights come from Torchvision; COCO AP uses pycocotools. The full detector was neither implemented nor trained from scratch. Code, tests and documentation were developed with AI assistance.

- [Selection](src/driving_data/selection.py), [training](src/driving_data/training.py), [later experiments](src/driving_data/failure_loop.py).
- [Architecture](docs/ARCHITECTURE.md): responsibilities of Parquet/DuckDB queries, SQLite recovery and single-machine Ray experiments.
- [Research history](docs/RESEARCH_INDEX.md): pilot, SmolVLM day/night checks, Ray timings and methods.

## Limitations

- A fixed third-party mirror of BDD100K validation data was resplit for this small local CPU study. It is not an official benchmark. Prefix grouping does not guarantee route or driver separation.
- Only 466,375 ROI predictor-head parameters are updated; the backbone and its statistics are frozen. No full-detector or VLM training.
- Equal image counts are not equal box counts, annotation time or labeling cost. No demonstrated labor saving, real-driving or deployment benefit.
- Ray is limited to single-machine preprocessing and was slower than serial processing for this small workload; it does not establish multi-machine production capability.
- Three training seeds and a small evaluation set limit statistical conclusions. Historical post-hoc matched-step ablations, failures and negative results remain available.

Code: [MIT](LICENSE). Data: [BDD100K terms](docs/BDD100K_LICENSE.rst).

[2026-09-19 maintenance](docs/maintenance/2026-09-19/README.md) · [2026-09-21 maintenance](docs/maintenance/2026-09-21/README.md)

2026-09-21: [Cache integrity and recovery](docs/maintenance/2026-09-21-cache/README.md).

## Contributing

[Guide](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Maintenance](docs/MAINTAINING.md)

[Report a bug](https://github.com/kimzclandi/ObjectDetectionDataSelection/issues/new?template=bug_report.yml) · [Request a feature](https://github.com/kimzclandi/ObjectDetectionDataSelection/issues/new?template=feature_request.yml)

## License

Code: [MIT](LICENSE). BDD100K data: [upstream terms](docs/BDD100K_LICENSE.rst). Model weights retain their upstream license.

[Naming and compatibility](docs/NAMING.md)
