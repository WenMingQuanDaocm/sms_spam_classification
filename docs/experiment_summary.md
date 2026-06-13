# 实验结果汇总

本文档整理已经真实运行得到的主要结果，供课程论文撰写时引用。所有数值均来自 `results/metrics/` 下的实验输出。

## 数据质量与清洗

- 原始样本数：5574
- 清洗后样本数：5160
- 删除完全重复样本：414
- 缺失值：0
- 空文本：0
- 非法标签：0
- 标签冲突：0

清洗后类别分布：

| Label | Count | Proportion |
|---|---:|---:|
| ham | 4518 | 0.8756 |
| spam | 642 | 0.1244 |

从 EDA 结果看，spam 短信平均字符数、平均单词数、数字数量和大写字母数量均高于 ham。该观察只作为数据描述和错误分析线索，不作为手工规则加入模型。

## 数据划分与 TF-IDF

使用 `random_state=42`，按 60%/20%/20% 分层划分：

| Split | Samples | Ham | Spam |
|---|---:|---:|---:|
| train | 3096 | 2711 | 385 |
| validation | 1032 | 903 | 129 |
| test | 1032 | 904 | 128 |

集合间重复文本数量：

| Pair | Overlap |
|---|---:|
| train-validation | 0 |
| train-test | 0 |
| validation-test | 0 |

TF-IDF 只在训练集上拟合，验证集和测试集只执行 `transform`。最终词表大小为 5000，三个 split 的特征维度均为 5000。

## 验证集模型表现

| Model | Accuracy | Macro-F1 | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.8750 | 0.4667 | 0.0000 | 0.0000 | 0.0000 |
| logistic regression | 0.9593 | 0.8914 | 1.0000 | 0.6744 | 0.8056 |
| MLP | 0.9864 | 0.9681 | 0.9752 | 0.9147 | 0.9440 |

多数类 baseline 的 Accuracy 较高，但 spam Recall 为 0，说明只看 Accuracy 会掩盖少数类失败。逻辑回归没有产生验证集误报，但漏报较多。MLP 在验证集上显著提高了 spam Recall 和 Macro-F1。

## MLP 默认训练

默认配置：

- Hidden layers: `(128, 64)`
- Dropout: `0.3`
- Optimizer: AdamW
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Batch size: `64`
- Early stopping patience: `8`

训练结果：

- Best epoch：5
- Epochs ran：13
- Validation Macro-F1：0.9681
- Validation confusion matrix：`[[900, 3], [11, 118]]`

训练曲线保存于 `figures/training/mlp_training_curves.png`。

## 超参数实验

学习率实验固定 Dropout=`0.3`：

| Learning Rate | Val Accuracy | Val Macro-F1 | Best Epoch |
|---:|---:|---:|---:|
| 0.01 | 0.9845 | 0.9633 | 5 |
| 0.001 | 0.9864 | 0.9681 | 5 |
| 0.0001 | 0.9864 | 0.9679 | 25 |

根据验证 Macro-F1，选择 `learning_rate=0.001`。

Dropout 实验固定 learning_rate=`0.001`：

| Dropout | Val Accuracy | Val Macro-F1 | Best Epoch |
|---:|---:|---:|---:|
| 0.0 | 0.9864 | 0.9681 | 3 |
| 0.3 | 0.9864 | 0.9681 | 5 |
| 0.5 | 0.9855 | 0.9660 | 5 |

`dropout=0.0` 和 `dropout=0.3` 验证 Macro-F1 持平。考虑默认方案和正则化，最终保留 `dropout=0.3`。

## 测试集最终评价

测试集只在模型选择和超参数实验完成后使用。

| Model | Accuracy | Macro-F1 | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.8760 | 0.4669 | 0.0000 | 0.0000 | 0.0000 |
| logistic regression | 0.9622 | 0.8996 | 1.0000 | 0.6953 | 0.8203 |
| MLP | 0.9845 | 0.9631 | 0.9746 | 0.8984 | 0.9350 |

测试集混淆矩阵：

| Model | Confusion Matrix |
|---|---|
| logistic regression | `[[904, 0], [39, 89]]` |
| MLP | `[[901, 3], [13, 115]]` |

最终结果表明，MLP 在测试集 Macro-F1 和 spam Recall 上明显优于逻辑回归，但逻辑回归保持了更强的 spam Precision 和可解释性。

## 错误分析输出

基于 MLP 测试集预测：

- False positives：3
- False negatives：13

输出文件：

- `results/error_analysis/false_positives.csv`
- `results/error_analysis/false_negatives.csv`

从样本看，一些 false positives 包含 “offer”“texts” 等容易与 spam 相关的词；一些 false negatives 具有玩笑文本、成人内容、变形拼写或上下文不明显等特点。这说明 TF-IDF + MLP 仍然可能受表面词形和上下文缺失限制。

## 可解释性线索

逻辑回归最支持 spam 的高权重特征包括：

- `call`
- `txt`
- `free`
- `text`
- `to`

这些特征符合垃圾短信中常见的呼叫、回复、免费奖励等模式，但高权重词也可能导致包含类似词的正常短信被误判。

## 论文结论要点

- 类别不均衡使多数类 baseline 的 Accuracy 看起来不低，但 Macro-F1 和 spam Recall 暴露了其无效性。
- TF-IDF + 逻辑回归已经能取得较好结果，且解释性强。
- MLP 在本次实验中提升了 spam Recall 和 Macro-F1，说明在固定 TF-IDF 特征上增加非线性建模有收益。
- MLP 并未彻底解决语义和上下文缺失问题，仍存在少量误报和漏报。
- 最终模型选择应综合性能、可解释性、训练成本和错误代价。
