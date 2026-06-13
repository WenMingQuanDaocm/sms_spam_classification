# 复现与审查记录

本文档记录正式实验环境、运行顺序、输出文件和数据泄漏审查结果。

## 环境信息

- OS：Windows-11-10.0.26200-SP0
- Python：3.14.5
- Python executable：`D:\app\Python314\python.exe`
- Training device：CPU
- CUDA available：false

主要依赖版本：

| Package | Version |
|---|---|
| joblib | 1.5.3 |
| matplotlib | 3.11.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| torch | 2.12.0 |
| tqdm | 4.68.2 |

这些版本已同步到 `requirements.txt`。

## 运行顺序

从已有原始数据文件开始，完整复现命令如下：

```powershell
python scripts/check_environment.py
python scripts/run_eda.py
python scripts/prepare_features.py
python scripts/run_classical_models.py
python scripts/train_mlp.py
python scripts/run_hyperparameter_analysis.py
python scripts/run_final_evaluation.py
python -m unittest discover -s tests -v
```

如果 Matplotlib 在当前 Windows 用户目录下没有缓存写权限，可在运行绘图脚本前设置：

```powershell
$env:MPLCONFIGDIR="$PWD\.matplotlib-cache"
```

该设置只影响 Matplotlib 缓存位置，不影响实验指标。

## 输入文件

原始数据文件：

```text
data/raw/SMSSpamCollection
```

该文件被 `.gitignore` 忽略，不应提交到 GitHub。

## 主要输出文件

数据与特征：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/tfidf_vectorizer.joblib
```

模型：

```text
models/logistic_regression/logistic_regression.joblib
models/mlp/best_model.pt
```

指标：

```text
results/metrics/eda_summary.json
results/metrics/preprocessing_summary.json
results/metrics/baseline_metrics.json
results/metrics/logistic_metrics.json
results/metrics/mlp_validation_metrics.json
results/metrics/learning_rate_experiments.csv
results/metrics/dropout_experiments.csv
results/metrics/baseline_test_metrics.json
results/metrics/logistic_test_metrics.json
results/metrics/mlp_test_metrics.json
results/metrics/model_comparison.csv
```

预测和错误分析：

```text
results/predictions/logistic_test_predictions.csv
results/predictions/mlp_test_predictions.csv
results/error_analysis/false_positives.csv
results/error_analysis/false_negatives.csv
```

图表：

```text
figures/eda/
figures/training/
figures/hyperparameters/
figures/confusion_matrix/
```

上述数据、模型、图表和结果产物均由 `.gitignore` 忽略，避免把原始数据或大文件提交到仓库。

## 数据泄漏审查

- 完全重复样本在划分前删除。
- 相同文本标签冲突数量为 0。
- 数据划分采用固定 `random_state=42` 和分层划分。
- 训练集、验证集、测试集之间重复文本数量均为 0。
- TF-IDF 只在训练集上 `fit`。
- 验证集和测试集只执行 `transform`。
- 超参数选择只使用验证集 Macro-F1。
- 测试集只在 `scripts/run_final_evaluation.py` 中用于最终评价。

## 当前限制

- 实验是在当前本机 Python 3.14.5 用户环境中完成，而非项目内虚拟环境。
- `requirements.txt` 固定的是项目直接依赖版本；底层传递依赖未完整冻结。
- Matplotlib 在默认用户缓存目录上存在权限警告，但图表生成成功。
- 实验结果来自单次固定随机种子运行；若论文需要稳定性统计，可扩展多随机种子重复实验，但这不属于当前项目设计要求。

## 最终审查状态

- 单元测试通过。
- 语法检查通过。
- 测试集未用于调参。
- 生成结果均来自真实脚本运行。
- 未手动修改指标、预测或图表。
