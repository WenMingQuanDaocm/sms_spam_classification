# AGENTS.md

## Project Overview

This repository implements a supervised-learning SMS spam classification course project.

The project uses the UCI SMS Spam Collection dataset and compares:

- Majority-class baseline
- Logistic regression
- A custom PyTorch MLP

The complete experiment pipeline is:

1. Load and inspect the dataset.
2. Clean invalid or duplicate samples.
3. Split the dataset into training, validation, and test sets.
4. Extract TF-IDF features.
5. Train and evaluate the baseline and logistic regression models.
6. Build and train a PyTorch MLP.
7. Perform controlled hyperparameter sensitivity experiments.
8. Generate figures, metrics, predictions, and error-analysis files.
9. Provide reproducible experimental results for the course paper.

Read `docs/project_plan.md` before making large or multi-file changes.

## Core Requirements

- Use Python.
- Use the UCI SMS Spam Collection dataset.
- Use `random_state=42` and fixed random seeds where applicable.
- Split the data into 60% training, 20% validation, and 20% test sets.
- Use stratified splitting.
- Remove exact duplicate samples before splitting.
- Check for missing values, empty text, invalid labels, and label conflicts.
- Fit TF-IDF only on the training set.
- Apply only `transform` to the validation and test sets.
- Never use the test set for hyperparameter selection.
- Use validation Macro-F1 as the primary model-selection metric.
- Evaluate final selected models on the test set only after tuning.
- Do not fabricate, modify, or manually improve experimental results.

## Required Models

The required models are:

1. `DummyClassifier` using the majority-class strategy.
2. `LogisticRegression` using TF-IDF features.
3. A custom MLP implemented with basic PyTorch modules.

The default MLP architecture is:

- TF-IDF input layer
- Fully connected layer with 128 hidden units
- ReLU
- Dropout
- Fully connected layer with 64 hidden units
- ReLU
- Dropout
- Two-class output layer

Do not replace the MLP with a pretrained language model or a fully encapsulated third-party classifier unless explicitly requested.

## Default Experiment Configuration

### TF-IDF

Use the following configuration as the initial baseline:

- `lowercase=True`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.98`
- `max_features=5000`
- `sublinear_tf=True`
- default L2 normalization

### Logistic Regression

Initial configuration:

- `C=1.0`
- `max_iter=2000`
- `random_state=42`

### MLP

Initial configuration:

- Hidden layers: `(128, 64)`
- Activation: ReLU
- Dropout: `0.3`
- Loss: `CrossEntropyLoss`
- Optimizer: AdamW
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Batch size: `64`
- Maximum epochs: `100`
- Early stopping patience: `8`

These are starting values, not guaranteed optimal values.

## Hyperparameter Experiments

The main controlled-variable experiments are:

### Learning rate

- `0.01`
- `0.001`
- `0.0001`

### Dropout

- `0.0`
- `0.3`
- `0.5`

Change only one experimental variable at a time.

Keep all other settings fixed and record:

- Complete configuration
- Best epoch
- Validation Accuracy
- Validation Macro-F1
- Training time
- Whether training was stable
- Any signs of overfitting or underfitting

Optional extension experiments may include:

- Hidden-layer structures
- Logistic regression `C`
- TF-IDF unigram versus unigram-plus-bigram features
- Class weighting
- Classification-threshold analysis

Do not expand the experiment scope without a clear reason.

## Evaluation Requirements

Report at least:

- Accuracy
- Macro-F1
- Spam Precision
- Spam Recall
- Spam F1
- Confusion matrix

Also generate and analyze:

- Training and validation Loss curves
- Training and validation Accuracy curves
- Hyperparameter sensitivity plots
- Representative false positives
- Representative false negatives
- Logistic regression feature weights or important words
- Model training time and model complexity where practical

The main goal is not only to obtain high scores, but also to explain:

- Why the models behave differently
- Whether MLP overfits
- Whether added model complexity is worthwhile
- Which messages are difficult to classify
- How class imbalance affects evaluation

## Project Organization

Use the following repository structure:

```text
sms_spam_classification/
├─AGENTS.md
├─README.md
├─requirements.txt
├─main.py
├─data/
│  ├─raw/
│  └─processed/
├─docs/
│  └─project_plan.md
├─src/
│  ├─config.py
│  ├─data_loader.py
│  ├─eda.py
│  ├─preprocess.py
│  ├─models.py
│  ├─train_baseline.py
│  ├─train_logistic.py
│  ├─train_mlp.py
│  ├─hyperparameter_analysis.py
│  ├─evaluate.py
│  └─utils.py
├─figures/
│  ├─eda/
│  ├─training/
│  ├─confusion_matrix/
│  └─hyperparameters/
├─models/
│  ├─logistic_regression/
│  └─mlp/
├─results/
│  ├─metrics/
│  ├─predictions/
│  └─error_analysis/
└─tests/
```

Do not place all project logic in one file.

## Coding Requirements

- Use clear English names for files, functions, classes, and variables.
- Add type hints to important functions.
- Keep modules focused on one responsibility.
- Avoid unnecessary abstractions and dependencies.
- Do not silently catch exceptions.
- Validate file paths and required dataset columns.
- Keep random-seed handling centralized.
- Preserve existing working behavior unless the current task requires a change.
- Add comments for important machine-learning decisions, but avoid excessive line-by-line comments.
- Save experiment outputs instead of relying only on terminal output.
- Use UTF-8 for text and CSV files where applicable.

## Reproducibility Rules

- Record the operating system, Python version, package versions, and training device.
- Save processed train, validation, and test splits.
- Save the fitted TF-IDF vectorizer.
- Save the best logistic regression model.
- Save the best MLP checkpoint.
- Save training history and hyperparameter experiment results.
- Generate `requirements.txt`.
- Use the same formal experiment environment for all results compared in the paper.
- Do not compare hyperparameter experiments generated on different computers or software environments unless this is explicitly studied.

## Verification

After making code changes:

1. Run the relevant script or test.
2. Report the exact command that was run.
3. Report whether it succeeded.
4. Do not claim success without running the check.
5. Summarize modified files.
6. Report remaining limitations or unresolved issues.
7. Do not overwrite valid experiment results without preserving or documenting them.

## Collaboration Rules

- For large or multi-file tasks, inspect the repository and propose a plan before editing.
- Do not rewrite unrelated files.
- Do not install new major dependencies without explaining why.
- Do not change the dataset split, metric definitions, or experiment design silently.
- Do not use the test set during tuning.
- Do not alter labels or predictions to improve scores.
- Do not fabricate experimental numbers, figures, logs, or conclusions.
- When results contradict expectations, preserve the real results and analyze them.
- When a requirement is ambiguous, prefer the simpler and more reproducible implementation.
