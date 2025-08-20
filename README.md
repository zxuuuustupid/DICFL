# Decoupling Before Composing (DBC)

## Abstract
Compositional Generalization is a challenge in research of AI, especially in **Compositional Zero-Shot Learning (CZSL)** tasks，model need to recognize attribute-object pairs that unseen in trainset. Therefore，we proposed a method **Decoupling Before Composing (DBC)**，aims to effectively decouple attributes and objects in visual representations and improve the model's generalisation ability on unseen combinations through feature reorganisation mechanisms.

This research method is not only applicable to traditional visual classification tasks, but can also be extended to **industrial applications**, such as high-speed train bearing fault detection, where it can still effectively identify potential fault categories even when only healthy samples are available.

---

## Method

### 1. Model Structure
DBC is composed by these core modules：

- **Feature Extracter**  
  Use ResNet-18 as an image feature extractor to output high-dimensional convolutional features.

- **Decoupler**  
  Map the global image features to the **attribute space** and **object space**, respectively.

- **Classifier**  
  Use a multi-layer perceptron (MLP) to classify attributes and objects separately.

- **Decoder**  
  Recombine attribute features with object features to generate new composite features to support reconstruction and enhancement.

---

### 2. Loss Function
The optimisation objectives of DBC consist of the following parts:

1. **Representation Loss**  
   Cross-entropy loss is used to predict attributes and objects, including positive and negative sample constraints.

2. **Masked Representation Loss**  
   Generate masks based on gradient differences to suppress the coupling characteristics between attributes and objects, ensuring the effectiveness of decoupling.

3. **Gradient Penalty**  
   Constrain the consistency of gradient distribution in different environments to reduce representation bias.

4. **Reconstruction Loss**  
   Use a decoder to reconstruct features and avoid loss of information.

5. **Residual Swap Loss**  
   Randomly swap attributes/object features and perform decoding to improve robustness to unseen combinations.

---

## Experiment

### 1. Dataset
We conducted experiments on three typical combination generalisation task datasets:

- **MIT-States**: An image dataset composed of attributes and objects.
- **UT-Zappos50K**: A shoe dataset containing a rich combination of attributes.
- **BJTU-RAO Bogie Dataset**: An industrial application dataset containing the health and fault status of high-speed train bearings under different loads.

### 2. Experiment Settings
- emb-dim：512  
- batch-size：32  
- lr：1e-4  
- opt：Adam  
- epoch：100  

The loss weights are set as follows:  

| Parameters             | Description    | Value   |
|------------------|----------------------|-----|
| `lambda_rep`     | Loss weight         | 1.0 |
| `lambda_grad`    | Gradient consistency weight | 1.0 |
| `lambda_rec`     | Reconstruction loss weight  | 1.0 |
| `lambda_res`     | Reorganisation exchange loss weight | 1.0 |
| `res_epoch`      | Epoch to start reorganisation training   | 1   |

---

### 3. Result
On **Combination zero-shot learning (CZSL)** task, DBC significantly improved the recognition accuracy of unseen combinations while maintaining stable performance for seen combinations.

实验结果表明：

- **MIT-States / UT-Zappos50K**  
  DBC significantly outperforms baseline methods (such as independent attribute-object classifiers) in terms of accuracy on unseen combinations.

- **BJTU-RAO Bogie Dataset**  
  Even when only healthy samples were used for training, DBC was still able to accurately identify fault modes such as **IR (inner ring fault) and OR (outer ring fault)**, achieving **fault detection without fault samples** in industrial applications.
---

## Conclusion
The **Disentangling Before Composing (DBC)** model proposed in this paper effectively improves the model's performance in **composition generalisation tasks** through **attribute-target decoupling**, **gradient consistency constraints**, and **feature reorganisation mechanisms**. Experimental results show that DBC demonstrates superior generalisation capabilities in both visual recognition and industrial application scenarios.

Future work can be further expanded in the following directions:
- Cross-modal extension (combined generalisation of image-text alignment)
- Real-time application of large-scale industrial monitoring data
- Combining generative models to further improve the robustness of unseen combinations

---

## Citation
If you use this code or draw on the research methods described in this article, please cite the following related work:

```bibtex
@article{DBC2025,
  title={Disentangling Before Composing: Attribute-Object Decomposition for Compositional Generalization},
  author={Tian Zhang et al.},
  journal={ArXiv preprint},
  year={2020}
}
