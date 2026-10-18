# Development workflow

| Branch | Scope |
|---|---|
| main | Integrated code after tests |
| demo/core | Dataset contracts, desktop and web foundations |
| demo/dataset-google | Google collection, query stages, Excel export |
| demo/train-yolo | Detection training and bounding-box validation |
| demo/train-quality | Quality training, source expansion, model comparisons |
| demo/inference-consensus | Three-source recognition and quality abstention |
| codex/tests-ci | Unit/integration tests and CI |

The branches contain complete project snapshots. Runtime modules stay together so changes can be tested before merging; branches are not separate partial applications. Use merge commits to retain feature history. Avoid committing images, checkpoints, credentials, local model registries and runtime logs.

Run `python -X utf8 -m unittest discover -s tests -q` before merging. GitHub Actions repeats these checks on Windows with CPU dependencies. Real-model evaluations require local weights and source datasets; they are not replaced by synthetic fixtures in CI.

Train detection independently:

```powershell
python scripts/train_yolo.py --release path/to/release --run model_runs/detector/new-run --epochs 20
```

Train source-labelled classification:

```powershell
python scripts/train_food9.py --base workspace/sic_merged --prepared --run model_runs/quality/new-run --registry models/candidate.json
```

Evaluate alternative algorithms with `scripts/benchmark_algorithms.py`. Fit confidence calibration using `scripts/calibrate_consensus.py`; evaluate all three recognition sources using `scripts/evaluate_consensus.py --full`. These scripts use existing local dataset/weight contracts described in the README, and do not download unknown model code.

Promote a model only after comparing on identical validation images and recording regressions. Report test scores after the decision. Keep the original checkpoints and point a new registry at immutable weights; never overwrite a live checkpoint to publish a different model.
