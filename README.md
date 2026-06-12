# SMS Spam Classification

This repository contains a supervised-learning course project for SMS spam
classification using the UCI SMS Spam Collection dataset.

The planned comparison includes:

- Majority-class baseline with `DummyClassifier`
- Logistic regression on TF-IDF features
- A custom PyTorch MLP on TF-IDF features

The project is intentionally staged. The current initialization phase only
creates the repository structure and configuration skeleton. It does not
download data, install dependencies, train models, or generate experimental
results.

## Data

Place the raw UCI SMS Spam Collection file at:

```text
data/raw/SMSSpamCollection
```

The expected raw format is:

```text
label<TAB>message
```

Valid labels are `ham` and `spam`.

## Experiment Rules

- Use `random_state=42` and fixed seeds where applicable.
- Remove exact duplicate samples before splitting.
- Use stratified 60% training, 20% validation, and 20% test splits.
- Fit TF-IDF only on the training split.
- Use validation Macro-F1 for model selection.
- Use the test split only for final evaluation after tuning.
- Preserve real experimental outputs; do not fabricate or manually improve
  metrics, predictions, or plots.

## Planned Structure

```text
sms_spam_classification/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
├─ figures/
│  ├─ eda/
│  ├─ training/
│  ├─ confusion_matrix/
│  └─ hyperparameters/
├─ models/
│  ├─ logistic_regression/
│  └─ mlp/
├─ results/
│  ├─ metrics/
│  ├─ predictions/
│  └─ error_analysis/
├─ scripts/
├─ src/
└─ tests/
```

## Environment

The initial dependency list is in `requirements.txt`. Exact frozen package
versions should be recorded after the project virtual environment is created.

To inspect the current Python environment without installing anything, run:

```powershell
python scripts/check_environment.py
```
