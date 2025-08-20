# Disentangling Before Composing (DBC)

## Abstract
组合泛化（Compositional Generalization）是人工智能研究中的关键挑战之一，特别是在 **组合零样本学习（Compositional Zero-Shot Learning, CZSL）** 任务中，模型需要识别训练集中未出现过的属性-目标组合。为此，我们提出了一种 **解耦-重组框架（Disentangling Before Composing, DBC）**，旨在在视觉表示中实现属性与目标的有效解耦，并通过特征重组机制提升模型在未见组合上的泛化能力。

本研究方法不仅适用于传统视觉分类任务，同时可扩展到 **工业应用场景**，例如高速列车轴承故障检测中，当仅有健康样本时，仍可实现对潜在故障类别的有效识别。

---

## Method

### 1. Model Structure
DBC 模型由以下几个核心模块组成：

- **Feature Extracter**  
  使用 ResNet-18 作为图像特征提取器，输出高维卷积特征。

- **Disentangler**  
  将全局图像特征分别映射到 **属性空间（attribute space）** 与 **目标空间（object space）**。

- **Classifier**  
  使用多层感知机（MLP）对属性与目标分别进行分类。

- **Decoder**  
  将属性特征与目标特征重新组合，生成新的复合特征，以支持重建与增强。

---

### 2. Loss Function
DBC 的优化目标由以下部分组成：

1. **Representation Loss**  
   对属性与目标的预测使用交叉熵损失，包含正负样本约束。

2. **Masked Representation Loss**  
   基于梯度差异生成掩码，抑制属性与目标的耦合特征，确保解耦有效性。

3. **Gradient Penalty**  
   在不同环境下约束梯度分布一致性，减少表示偏差。

4. **Reconstruction Loss**  
   利用解码器对特征进行重建，避免表示信息丢失。

5. **Residual Swap Loss**  
   随机交换属性/目标特征并进行解码，提升未见组合的鲁棒性。

---

## Experiment

### 1. Dataset
我们在三个典型的组合泛化任务数据集上进行了实验：

- **MIT-States**：由属性与物体组合而成的图像数据集。  
- **UT-Zappos50K**：鞋类数据集，包含丰富的属性组合。  
- **BJTU-RAO Bogie Dataset**：工业应用数据集，包含高速列车轴承在不同载荷下的健康与故障状态。

### 2. Experiment Settings
- emb-dim：512  
- batch-size：32  
- lr：1e-4  
- opt：Adam  
- epoch：100  

各损失权重设置如下：  

| 参数             | 说明                 | 值   |
|------------------|----------------------|-----|
| `lambda_rep`     | 表示损失权重         | 1.0 |
| `lambda_grad`    | 梯度一致性权重       | 1.0 |
| `lambda_rec`     | 重建损失权重         | 1.0 |
| `lambda_res`     | 重组交换损失权重     | 1.0 |
| `res_epoch`      | 启动重组训练的轮次   | 1   |

---

### 3. Result
在 **组合零样本学习（CZSL）** 任务上，DBC 显著提升了未见组合的识别准确率，同时在见过组合的性能保持稳定。  

实验结果表明：

- **MIT-States / UT-Zappos50K**  
  DBC 在未见组合上的准确率显著超过基线方法（如独立属性-物体分类器）。  

- **BJTU-RAO Bogie Dataset**  
  在仅有健康样本参与训练的情况下，DBC 依然能够准确识别 **IR（内圈故障）、OR（外圈故障）** 等故障模式，实现了工业应用场景中的 **无故障样本故障检测**。

---

## Conclusion
本文提出的 **Disentangling Before Composing (DBC)** 模型通过 **属性-目标解耦**、**梯度一致性约束** 和 **特征重组机制**，有效提升了模型在 **组合泛化任务** 中的表现。实验结果表明，DBC 在视觉识别和工业应用场景中均展现出优越的泛化能力。

未来工作可在以下方向进一步拓展：
- 跨模态扩展（图像-文本对齐的组合泛化）
- 工业大规模监测数据的实时应用
- 结合生成模型进一步提升未见组合的鲁棒性

---

## Citation
如果您使用了该代码或借鉴了本文的研究方法，请引用以下相关工作：

```bibtex
@article{DBC2025,
  title={Disentangling Before Composing: Attribute-Object Decomposition for Compositional Generalization},
  author={Tian Zhang et al.},
  journal={ArXiv preprint},
  year={2020}
}
