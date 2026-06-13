# 垃圾短信分类项目

本仓库用于完成一个监督学习课程项目：基于 UCI SMS Spam Collection 数据集进行短信垃圾分类。

比较的模型包括：

- 使用 `DummyClassifier` 的多数类基线模型
- 基于 TF-IDF 特征的逻辑回归模型
- 基于 TF-IDF 特征的自定义 PyTorch MLP 模型

项目按阶段推进。当前代码已覆盖数据读取、EDA、数据划分、TF-IDF、baseline、逻辑回归、MLP、超参数实验、最终测试评价和复现整理。

阶段性实验结果汇总见：

```text
docs/experiment_summary.md
docs/reproducibility_check.md
```

## 数据

请将 UCI SMS Spam Collection 原始文件放在：

```text
data/raw/SMSSpamCollection
```

原始文件格式应为：

```text
label<TAB>message
```

合法标签只有 `ham` 和 `spam`。

## 实验规则

- 使用 `random_state=42`，并在适用位置固定随机种子。
- 在数据划分前删除完全重复样本。
- 使用分层划分，比例固定为 60% 训练集、20% 验证集、20% 测试集。
- TF-IDF 只能在训练集上拟合。
- 模型选择使用验证集 Macro-F1。
- 测试集只能在调参完成后用于最终评估。
- 保留真实实验输出，不得伪造或手动改善指标、预测结果或图表。

## 环境

依赖版本已固定在 `requirements.txt`。检查当前 Python 环境：

```powershell
python scripts/check_environment.py
```

## 运行顺序

```powershell
python scripts/run_eda.py
python scripts/prepare_features.py
python scripts/run_classical_models.py
python scripts/train_mlp.py
python scripts/run_hyperparameter_analysis.py
python scripts/run_final_evaluation.py
python -m unittest discover -s tests -v
```

## 第二阶段：数据读取与 EDA

```powershell
python scripts/run_eda.py
```

读取 `data/raw/SMSSpamCollection`，检查缺失值、空文本、非法标签、重复样本和标签冲突，然后生成 EDA 图表与 `results/metrics/eda_summary.json`。该脚本不会划分数据、拟合 TF-IDF、训练模型或在测试集上评估。

## 第三阶段：数据划分与 TF-IDF

```powershell
python scripts/prepare_features.py
```

执行保守清洗，按 60%/20%/20% 分层划分训练集、验证集和测试集，并保存：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/tfidf_vectorizer.joblib
results/metrics/preprocessing_summary.json
```

TF-IDF 只在训练集上拟合，验证集和测试集只执行 `transform`。

## 第四阶段：Baseline 与逻辑回归

```powershell
python scripts/run_classical_models.py
```

只使用训练集和验证集：

- 训练多数类 baseline，并在验证集上评价。
- 使用已保存的训练集 TF-IDF 向量器转换训练集和验证集。
- 训练默认逻辑回归，并在验证集上评价。
- 保存逻辑回归模型和特征权重。

该阶段不会读取或评价测试集。

## 第五阶段：MLP 基础训练

```powershell
python scripts/train_mlp.py
```

只使用训练集和验证集：

- 加载训练集拟合好的 TF-IDF 向量器。
- 将训练集和验证集 TF-IDF 特征转换为 `float32` 张量。
- 训练默认 MLP：`5000 -> 128 -> 64 -> 2`。
- 使用 AdamW、CrossEntropyLoss 和 Early Stopping。
- 按验证集 Macro-F1 保存最佳 checkpoint。

输出包括：

```text
models/mlp/best_model.pt
results/metrics/mlp_training_history.csv
results/metrics/mlp_validation_metrics.json
figures/training/mlp_training_curves.png
```

该阶段不会读取或评价测试集。

## 第六阶段：超参数敏感性实验

```powershell
python scripts/run_hyperparameter_analysis.py
```

执行两个受控实验：

- 学习率实验：固定 Dropout 为 `0.3`，只比较 `0.01`、`0.001`、`0.0001`。
- Dropout 实验：固定学习率为上一组验证 Macro-F1 最优值，只比较 `0.0`、`0.3`、`0.5`。

输出包括：

```text
results/metrics/learning_rate_experiments.csv
results/metrics/dropout_experiments.csv
figures/hyperparameters/learning_rate_sensitivity.png
figures/hyperparameters/dropout_sensitivity.png
```

每个实验还会保存独立的 checkpoint、训练历史、验证指标和曲线，避免覆盖第五阶段默认 MLP 结果。该阶段不会读取或评价测试集。

## 第七阶段：最终测试与错误分析

```powershell
python scripts/run_final_evaluation.py
```

完成验证集调参后，首次正式使用测试集：

- 评价多数类 baseline、逻辑回归和 MLP。
- 保存测试集指标和模型对比表。
- 保存逻辑回归和 MLP 的测试集预测。
- 生成逻辑回归和 MLP 的混淆矩阵图。
- 基于 MLP 预测导出误报和漏报样本。

输出包括：

```text
results/metrics/baseline_test_metrics.json
results/metrics/logistic_test_metrics.json
results/metrics/mlp_test_metrics.json
results/metrics/model_comparison.csv
results/predictions/logistic_test_predictions.csv
results/predictions/mlp_test_predictions.csv
results/error_analysis/false_positives.csv
results/error_analysis/false_negatives.csv
figures/confusion_matrix/logistic_confusion_matrix.png
figures/confusion_matrix/mlp_confusion_matrix.png
```

## 项目结构

```text
sms_spam_classification/
├─ data/
├─ docs/
├─ figures/
├─ models/
├─ results/
├─ scripts/
├─ src/
└─ tests/
```
