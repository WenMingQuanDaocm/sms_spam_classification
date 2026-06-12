# 垃圾短信分类项目完整实施方案

## 1.项目名称

**基于TF-IDF与监督学习的垃圾短信分类研究——逻辑回归与多层感知机的对比分析**

## 2.项目目标

本项目以UCI SMS Spam Collection数据集为基础，将英文短信划分为：

- `ham`：正常短信
- `spam`：垃圾短信

项目需要完成从数据获取、数据探索、特征提取、模型训练、超参数分析到测试评价的完整监督学习实验流程。

核心研究问题包括：

1.基于TF-IDF的文本特征能否有效识别垃圾短信？
2.逻辑回归与MLP在短信分类中的性能有何差异？
3.模型复杂度增加后，性能是否获得明显提升？
4.学习率和Dropout对MLP训练稳定性与泛化性能有何影响？
5.哪些短信最容易被误判？
6.类别不均衡会如何影响Accuracy和Macro-F1？
7.逻辑回归能够学习到哪些有代表性的垃圾短信关键词？

项目重点不是单纯追求最高准确率，而是完成规范、可复现、可解释的实验，并形成有依据的技术分析。

## 3.数据集

### 3.1数据来源

使用UCI Machine Learning Repository提供的SMS Spam Collection数据集。

原始数据文件建议放置为：

```text
data/raw/SMSSpamCollection
```

原始数据每行包含：

```text
label<TAB>message
```

示例：

```text
ham	Are you coming home tonight?
spam	URGENT! You have won a cash prize...
```

### 3.2数据字段

| 字段 | 含义 |
|---|---|
| `label` | 原始类别标签，取值为`ham`或`spam` |
| `message` | 短信原文 |
| `target` | 数值标签，`ham=0`，`spam=1` |

### 3.3数据检查

必须检查：

- 数据规模
- 类别分布
- 缺失值
- 空文本
- 完全重复样本
- 相同文本是否存在标签冲突
- 非法标签
- 异常编码

完全重复样本应在划分数据集前删除，以避免同一短信同时进入训练集和测试集。

不得擅自删除其他样本。任何额外清洗都应说明理由。

## 4.软件和硬件准备

### 4.1推荐环境

- 操作系统：Windows11 64位
- 开发工具：Visual Studio Code
- Python：以项目虚拟环境中的实际版本为准
- 训练设备：ThinkBook14+2026酷睿版CPU
- GPU：Intel集成显卡不参与常规PyTorch CUDA训练
- CUDA：未使用

### 4.2主要依赖

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- tqdm
- joblib

### 4.3虚拟环境

在项目根目录中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch tqdm joblib
pip freeze > requirements.txt
```

### 4.4环境信息记录

项目应记录：

- OS
- Python版本
- PyTorch版本
- scikit-learn版本
- pandas版本
- NumPy版本
- Matplotlib版本
- CUDA是否可用
- 实际训练设备

## 5.建议目录结构

```text
sms_spam_classification/
├─AGENTS.md
├─README.md
├─requirements.txt
├─main.py
├─data/
│  ├─raw/
│  │  └─SMSSpamCollection
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

## 6.数据探索方案

### 6.1基础统计

至少输出：

- 前若干条数据
- 数据集形状
- 每列类型
- 缺失值数量
- 各类别数量与比例
- 重复样本数量
- 标签冲突数量

### 6.2辅助统计特征

为每条短信构造仅用于探索和解释的统计字段：

- `char_count`：字符数量
- `word_count`：单词数量
- `digit_count`：数字数量
- `exclamation_count`：感叹号数量
- `uppercase_count`：大写字母数量

### 6.3建议图表

至少生成：

1.类别分布柱状图
2.ham与spam字符长度分布图
3.ham与spam单词数量分布图
4.数字数量或感叹号数量分布图

所有图保存到：

```text
figures/eda/
```

图表生成后需要结合真实数据分析，不得预设垃圾短信一定更长或一定包含更多感叹号。

## 7.文本处理原则

### 7.1允许的基础处理

- 删除文本首尾空白
- 删除空字符串
- 将标签映射为数值
- 由TF-IDF统一转换为小写

### 7.2不建议直接删除的内容

- 数字
- 电话号码
- URL
- 货币符号
- 感叹号
- 大写词
- 拼写变形

这些内容可能是垃圾短信的重要识别线索。

### 7.3数据泄漏控制

必须先划分数据，再拟合TF-IDF。

正确流程：

```text
原始数据
→清洗与去重
→训练/验证/测试划分
→在训练集上fit TF-IDF
→验证集和测试集只transform
```

禁止先对全部数据拟合TF-IDF后再划分。

## 8.数据集划分

固定为：

- 训练集：60%
- 验证集：20%
- 测试集：20%
- 分层划分：`stratify=y`
- 随机种子：`42`

划分后必须检查：

- 三个集合的样本数量
- 三个集合的类别比例
- 集合间是否存在重复文本
- 测试集是否被训练或调参代码访问

划分结果保存为：

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

## 9.TF-IDF特征提取

### 9.1初始配置

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=5000,
    sublinear_tf=True
)
```

### 9.2参数含义

- `ngram_range=(1,2)`：使用单词和双词短语
- `min_df=2`：删除只出现一次的极低频词
- `max_df=0.98`：过滤几乎所有短信中都出现的词
- `max_features=5000`：限制特征维度
- `sublinear_tf=True`：减弱重复词频的过强影响
- 默认L2归一化：降低短信长度差异造成的尺度影响

### 9.3保存结果

保存拟合后的向量器：

```text
models/tfidf_vectorizer.joblib
```

逻辑回归可直接使用稀疏矩阵。

MLP输入需要转换为`float32`稠密矩阵。若内存占用过高，可将`max_features`降至3000，但需要记录并说明修改原因。

## 10.模型设计

### 10.1多数类Baseline

使用：

```python
DummyClassifier(strategy="most_frequent")
```

作用：

- 建立最低性能参考
- 说明Accuracy在类别不均衡任务中的局限
- 验证真实模型是否学到了有效特征

应报告其Accuracy、Macro-F1和spam Recall。

### 10.2逻辑回归

初始配置：

```python
LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=42
)
```

作用：

- 作为传统线性分类模型
- 与MLP比较性能、训练时间和可解释性
- 提取高权重词语
- 分析高维稀疏TF-IDF是否已经近似线性可分

可选参数实验：

```text
C=0.01、0.1、1、10、100
```

### 10.3多层感知机MLP

默认结构：

```text
输入层：TF-IDF特征维度
隐藏层1：128
激活函数：ReLU
Dropout：0.3
隐藏层2：64
激活函数：ReLU
Dropout：0.3
输出层：2
```

训练配置：

- 损失函数：CrossEntropyLoss
- 优化器：AdamW
- 学习率：0.001
- 权重衰减：1e-4
- Batch Size：64
- 最大Epoch：100
- Early Stopping patience：8
- 主要模型选择指标：验证集Macro-F1

每个Epoch至少记录：

- Train Loss
- Validation Loss
- Train Accuracy
- Validation Accuracy
- Validation Macro-F1
- 当前学习率
- 是否保存最佳模型

最佳模型保存为：

```text
models/mlp/best_model.pt
```

## 11.超参数敏感性实验

课程要求至少分析两个超参数，并采用控制变量法。

### 11.1学习率实验

固定：

- 隐藏层结构`(128,64)`
- Dropout`0.3`
- Batch Size`64`
- Weight Decay`1e-4`

只改变：

```text
0.01
0.001
0.0001
```

记录：

- 最佳Epoch
- Val Accuracy
- Val Macro-F1
- 收敛速度
- Loss曲线稳定性
- 是否发生振荡或未充分收敛

### 11.2Dropout实验

固定学习率为上一阶段选出的最优值，只改变：

```text
0.0
0.3
0.5
```

分析：

- 不使用Dropout时是否过拟合
- 适中Dropout是否提高泛化能力
- Dropout过大是否导致欠拟合
- Train/Val曲线差距如何变化

### 11.3可选扩展

时间允许时，可以比较：

```text
隐藏层结构：
(64,)
(128,64)
(256,128,64)
```

或比较：

```text
TF-IDF：
(1,1)
(1,2)
```

不得一次改变多个变量后直接归因于其中某一个变量。

## 12.评价指标

必须报告：

- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- 混淆矩阵

重点关注spam Precision、spam Recall和spam F1。

不得仅凭Accuracy评价模型。

## 13.必须生成的图表

建议最终至少包含：

| 图表 | 内容 |
|---|---|
| 图1 | 类别分布柱状图 |
| 图2 | 不同类别短信长度分布 |
| 图3 | MLP训练集和验证集Loss曲线 |
| 图4 | MLP训练集和验证集Accuracy曲线 |
| 图5 | 学习率敏感性图 |
| 图6 | Dropout敏感性图 |
| 图7 | 逻辑回归混淆矩阵 |
| 图8 | MLP混淆矩阵 |
| 图9 | 模型性能对比图 |

课程要求的Loss和Accuracy可绘制为同一张双面板图。

所有图表必须由真实实验结果生成，并在正文中进行分析。

## 14.错误分析

最终测试后分别保存：

```text
results/error_analysis/false_positives.csv
results/error_analysis/false_negatives.csv
```

每条记录至少包含：

- `message`
- `true_label`
- `predicted_label`
- `spam_probability`

分别分析5至10条有代表性的误报和漏报。

可从以下角度分析：

- 正常短信中出现垃圾短信高权重词
- 垃圾短信表达隐晦
- 短信过短
- 拼写变形
- TF-IDF缺少上下文语义
- 训练数据中类似表达不足
- 分类阈值造成的Precision与Recall权衡

结论应使用“可能”“表明”“从结果看”等谨慎表达，不应在没有证据时作绝对判断。

## 15.逻辑回归可解释性分析

提取逻辑回归权重：

- 最支持spam判断的前20个特征
- 最支持ham判断的前20个特征

保存为：

```text
results/metrics/logistic_feature_weights.csv
```

分析：

- 高权重词是否符合垃圾短信常见模式
- 双词短语是否比单词更有解释力
- 是否存在可能导致误判的词语
- 线性模型为何可能已经取得较好性能

## 16.推荐实验执行顺序

### 第一阶段：项目初始化

1.创建目录结构
2.创建虚拟环境
3.安装依赖
4.生成`requirements.txt`
5.创建环境检查脚本
6.创建`.gitignore`
7.确认数据文件路径

### 第二阶段：数据读取与EDA

1.读取数据
2.检查缺失值、空文本和非法标签
3.检查重复样本与标签冲突
4.完成基础统计
5.生成EDA图表
6.验证清洗后的数据规模

### 第三阶段：数据划分与TF-IDF

1.完成60%/20%/20%分层划分
2.保存三个数据集
3.仅在训练集拟合TF-IDF
4.转换验证集和测试集
5.保存向量器
6.检查数据泄漏和集合重叠

### 第四阶段：Baseline与逻辑回归

1.训练多数类Baseline
2.训练默认逻辑回归
3.在验证集上评价
4.必要时调整`C`
5.保存模型
6.提取重要词语

### 第五阶段：MLP基础实现

1.实现Dataset和DataLoader
2.实现MLP结构
3.实现训练循环
4.实现验证循环
5.实现Early Stopping
6.保存最佳模型
7.绘制训练曲线

### 第六阶段：超参数实验

1.完成学习率实验
2.选择最佳学习率
3.完成Dropout实验
4.生成实验表格
5.绘制敏感性图
6.确定最终配置

### 第七阶段：最终测试

1.固定全部参数
2.加载最佳模型
3.只进行正式测试集评价
4.生成混淆矩阵
5.导出误报和漏报
6.生成最终模型对比表

### 第八阶段：论文整理

1.整理环境配置
2.整理数据表和图
3.整理模型结构与参数
4.撰写结果分析
5.撰写技术综述
6.撰写MLP深入分析
7.完成摘要、引言、总结和参考文献

## 17.论文技术分析要点

### 17.1TF-IDF

应说明：

- TF
- IDF
- TF-IDF
- 平滑IDF
- 对数词频
- L2归一化
- 单词与双词特征

### 17.2逻辑回归

应说明：

- 线性打分
- Sigmoid
- 二元交叉熵
- L2正则化
- 参数`C`
- 特征权重可解释性

### 17.3MLP深入分析

MLP作为主要深入分析技术，应说明：

- 全连接层
- ReLU
- Dropout
- Softmax
- 交叉熵损失
- 前向传播
- 反向传播
- AdamW
- Weight Decay
- Early Stopping

### 17.4评价指标公式

应说明：

- Accuracy
- Precision
- Recall
- F1
- Macro-F1

每个公式需要结合项目说明其实际意义。

## 18.预期可形成的研究见解

以下内容只是待验证假设，最终必须以真实实验结果为准。

### 18.1复杂模型不一定更好

TF-IDF高维稀疏特征可能已经接近线性可分，逻辑回归可能与MLP性能接近，甚至表现更好。

### 18.2MLP的能力受特征表示限制

MLP具有非线性表达能力，但TF-IDF缺少完整语序和上下文语义。更复杂的模型无法恢复输入中已经丢失的信息。

### 18.3MLP可能过拟合

当输入为5000维，第一隐藏层为128时，第一层就包含约64万个权重，而数据集只有约5500条短信，因此可能出现训练集表现持续提高、验证集性能停滞的现象。

### 18.4Dropout存在平衡

Dropout太小可能过拟合，太大可能欠拟合，适中取值可能在特征保留与正则化之间取得平衡。

### 18.5学习率影响稳定性和收敛速度

不能只比较最终Accuracy，还应比较最佳Epoch、Loss波动和训练稳定性。

### 18.6Accuracy可能掩盖少数类失败

多数类Baseline可能获得较高Accuracy，但spam Recall可能接近0，因此必须使用Macro-F1和混淆矩阵。

### 18.7误报与漏报代价不同

误报会拦截正常短信，漏报会放过垃圾短信。最优阈值取决于实际场景中的错误代价。

### 18.8模型选择需要综合考虑

除性能外，还需要比较：

- 可解释性
- 训练时间
- 参数数量
- 部署成本
- 错误代价
- 结果稳定性

## 19.结果文件规划

### 指标

```text
results/metrics/baseline_metrics.json
results/metrics/logistic_metrics.json
results/metrics/mlp_metrics.json
results/metrics/model_comparison.csv
results/metrics/logistic_feature_weights.csv
```

### 训练记录

```text
results/metrics/mlp_training_history.csv
results/metrics/learning_rate_experiments.csv
results/metrics/dropout_experiments.csv
```

### 预测

```text
results/predictions/logistic_test_predictions.csv
results/predictions/mlp_test_predictions.csv
```

### 错误分析

```text
results/error_analysis/false_positives.csv
results/error_analysis/false_negatives.csv
```

## 20.复现性要求

项目应做到：

- 固定随机种子
- 保存数据划分
- 保存TF-IDF向量器
- 保存最佳模型
- 保存训练历史
- 保存超参数实验记录
- 保存依赖版本
- 记录OS、Python、GPU和CUDA信息
- 正式对比实验尽量在同一台电脑和同一环境中完成

开发和调试可以在不同电脑上进行，但论文中的正式超参数实验和最终结果应尽量来自同一台设备和同一软件环境。

## 21.风险与处理

### MLP过拟合

可尝试：

- 增大Dropout
- 增大Weight Decay
- 减少隐藏层规模
- 使用Early Stopping

### spam Recall过低

可尝试：

- 逻辑回归使用`class_weight="balanced"`
- MLP使用类别加权交叉熵
- 使用Macro-F1选择模型
- 扩展进行分类阈值分析

### 内存占用过高

可尝试：

- 将`max_features`从5000降至3000
- 使用`float32`
- 及时释放不再使用的稠密矩阵
- 避免保存多份重复特征矩阵

### 实验结果不稳定

需要：

- 固定Python、NumPy和PyTorch随机种子
- 固定DataLoader生成器
- 固定数据划分
- 统一软件版本
- 正式实验使用同一设备

### 逻辑回归优于MLP

不得修改或隐藏结果。应分析：

- 数据规模有限
- TF-IDF适合线性分类
- MLP参数过多
- 非线性模型未带来相应收益
- 简单模型可能具有更高应用价值

## 22.最终验收清单

项目完成时应具备：

- 原始数据文件
- 清洗后的数据
- 固定训练集、验证集和测试集
- EDA脚本和图表
- 训练集拟合的TF-IDF
- 多数类Baseline
- 逻辑回归模型
- PyTorch MLP
- 至少两个超参数敏感性实验
- 训练和验证曲线
- 混淆矩阵
- Accuracy、Macro-F1等指标
- 逻辑回归重要关键词
- 误报和漏报案例
- 模型和向量器文件
- `requirements.txt`
- 环境信息
- 可复现运行说明
- 支持课程论文撰写的完整实验结果

## 23.Codex分阶段执行原则

不要要求Codex一次性完成整个项目。

建议按以下阶段逐步发送任务：

1.项目初始化
2.数据读取和EDA
3.数据划分与TF-IDF
4.Baseline和逻辑回归
5.MLP结构与训练循环
6.超参数实验
7.最终评价与错误分析
8.代码审查和复现检查

每个阶段完成后都应：

- 检查修改文件
- 运行相关命令
- 查看真实输出
- 确认不存在数据泄漏
- 再进入下一阶段
