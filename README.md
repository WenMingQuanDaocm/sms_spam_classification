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
