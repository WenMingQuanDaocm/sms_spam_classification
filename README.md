# 垃圾短信分类项目

本仓库用于完成一个监督学习课程项目：基于 UCI SMS Spam Collection 数据集进行短信垃圾分类。

计划比较的模型包括：

- 使用 `DummyClassifier` 的多数类基线模型
- 基于 TF-IDF 特征的逻辑回归模型
- 基于 TF-IDF 特征的自定义 PyTorch MLP 模型

项目按阶段推进。当前初始化阶段只创建项目结构、配置骨架和基础脚本；不会下载数据、安装依赖、训练模型或生成实验结果。

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

## 项目结构

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

## 环境检查

初始依赖清单位于 `requirements.txt`。正式实验环境创建完成后，应记录精确冻结版本。

不安装任何依赖的情况下，可以运行以下命令检查当前 Python 环境：

```powershell
python scripts/check_environment.py
```

## 第二阶段：数据读取与 EDA

当原始数据和依赖都准备好后，运行：

```powershell
python scripts/run_eda.py
```

该脚本会读取 `data/raw/SMSSpamCollection`，检查缺失值、空文本、非法标签、重复样本和标签冲突，然后生成 EDA 图表与 `results/metrics/eda_summary.json`。

该脚本不会划分数据、拟合 TF-IDF、训练模型或在测试集上评估。

## 第三阶段：数据划分与 TF-IDF

当原始数据和依赖都准备好后，运行：

```powershell
python scripts/prepare_features.py
```

该脚本会先执行保守清洗，然后按 60%/20%/20% 进行分层训练、验证、测试划分，保存到：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

随后脚本只在训练集上拟合 TF-IDF，并将验证集和测试集仅用于 `transform`。拟合后的向量器保存为：

```text
models/tfidf_vectorizer.joblib
```

预处理摘要保存为：

```text
results/metrics/preprocessing_summary.json
```

## 第四阶段：Baseline 与逻辑回归

在第二、第三阶段完成后，运行：

```powershell
python scripts/run_classical_models.py
```

该脚本只使用训练集和验证集：

- 训练多数类 baseline，并在验证集上评价。
- 使用已保存的训练集 TF-IDF 向量器转换训练集和验证集。
- 训练默认逻辑回归，并在验证集上评价。
- 保存逻辑回归模型和特征权重。

输出文件包括：

```text
results/metrics/baseline_metrics.json
results/metrics/logistic_metrics.json
results/metrics/logistic_feature_weights.csv
models/logistic_regression/logistic_regression.joblib
```

该阶段不会读取或评价测试集。

## 第六阶段：超参数敏感性实验

在第五阶段完成后，运行：

```powershell
python scripts/run_hyperparameter_analysis.py
```

该脚本执行两个受控实验：

- 学习率实验：固定 Dropout 为 `0.3`，只比较 `0.01`、`0.001`、`0.0001`。
- Dropout 实验：固定学习率为上一组验证 Macro-F1 最优值，只比较 `0.0`、`0.3`、`0.5`。

输出文件包括：

```text
results/metrics/learning_rate_experiments.csv
results/metrics/dropout_experiments.csv
figures/hyperparameters/learning_rate_sensitivity.png
figures/hyperparameters/dropout_sensitivity.png
```

每个实验还会保存独立的 checkpoint、训练历史、验证指标和曲线，避免覆盖第五阶段默认 MLP 结果。该阶段仍然不会读取或评价测试集。

## 第五阶段：MLP 基础训练

在第三阶段完成后，运行：

```powershell
python scripts/train_mlp.py
```

该脚本只使用训练集和验证集：

- 加载训练集拟合好的 TF-IDF 向量器。
- 将训练集和验证集 TF-IDF 特征转换为 `float32` 张量。
- 训练默认 MLP：`5000 -> 128 -> 64 -> 2`。
- 使用 AdamW、CrossEntropyLoss 和 Early Stopping。
- 按验证集 Macro-F1 保存最佳 checkpoint。

输出文件包括：

```text
models/mlp/best_model.pt
results/metrics/mlp_training_history.csv
results/metrics/mlp_validation_metrics.json
figures/training/mlp_training_curves.png
```

该阶段不会读取或评价测试集。
