# Vision Transformer Patch Size Comparison

## 📌 Project Overview

This project implements a **Vision Transformer (ViT)** for image classification and studies how different image patch sizes affect:

- Classification accuracy
- Image representation
- Number of image patches
- Model computation
- Training time

Three different patch sizes are compared:

- 4 × 4
- 8 × 8
- 16 × 16

The experiment uses the **CIFAR-10 dataset** and a TensorFlow/Keras-based Vision Transformer.

---

## 🎯 Objective

The main objective of this project is to develop a Vision Transformer using different image patch sizes and analyze how patch size affects image representation and computational requirements.

Smaller patches preserve more detailed spatial information but create more tokens for the Transformer to process.

Larger patches reduce the number of tokens and computational requirements but may lose fine-grained image information.

---

## 📊 Dataset

The project uses the **CIFAR-10 dataset**.

CIFAR-10 contains:

- 60,000 color images
- Image size: 32 × 32 pixels
- 10 image classes
- RGB images

The dataset is divided into training and testing sets.

---

## 🧠 Vision Transformer Architecture

The Vision Transformer follows these major steps:

```text
Input Image
     ↓
Divide Image into Patches
     ↓
Flatten Patches
     ↓
Linear Patch Projection
     ↓
Add Class Token
     ↓
Add Positional Embedding
     ↓
Transformer Encoder
     ↓
Layer Normalization
     ↓
Multi-Head Self Attention
     ↓
MLP / Feed Forward Network
     ↓
Classification Head
     ↓
Class Prediction
```
