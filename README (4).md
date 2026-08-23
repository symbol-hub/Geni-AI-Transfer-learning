# Image Classification using CNN and Transfer Learning

Image classification pipeline comparing a Convolutional Neural Network (CNN) trained from scratch against MobileNetV2 (transfer learning), evaluated on the same dataset with the same metrics.

## Overview

This notebook answers a simple question: if you want a model to recognise what's in a picture, is it better to train one from zero, or to start from a model that has already learned to see?

It builds and trains two models on the same data:

- **Model A — CNN from scratch:** A small custom CNN (3 conv blocks) that learns every feature from the training data alone.
- **Model B — MobileNetV2 (transfer learning):** A model pre-trained on 1.2 million ImageNet images, adapted to the target classes via a frozen-head training stage followed by fine-tuning of the last ~30 layers.

Both models are then evaluated and compared using the same metrics and test set.

## Reference paper

- Kornblith, S., Shlens, J., & Le, Q. V. (2019). *Do Better ImageNet Models Transfer Better?* CVPR 2019. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 colour images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 50,000 training images, 10,000 test images. Loaded directly via `tensorflow.keras.datasets.cifar10` — no manual download needed.

## Notebook structure

1. **Research Paper Selection** — papers and dataset referenced, with links.
2. **Task 1: Dataset Preparation** — load, inspect, normalize, split, augment.
3. **Task 2: Model Implementation and Fine-tuning**
   - Build and train the CNN from scratch.
   - Build MobileNetV2 transfer model, train the head, then fine-tune top layers.
   - Feature map visualization.
4. **Task 3: Model Evaluation and Performance Comparison** — accuracy, precision, recall, F1-score, confusion matrices, training curves, summary table.
5. **Training Time and Model Size Comparison**
6. **Prediction Demonstration** — sample test images with actual vs. predicted labels and confidence.
7. **Research Paper Comparison** — how this notebook's setup differs from the reference paper's.
8. **Advantages and Limitations** of each approach.
9. **Conclusion** — result interpretation and comparison against the paper's reported numbers.

## Requirements

- Python 3
- TensorFlow / Keras
- NumPy, Matplotlib, Pandas, Seaborn
- scikit-learn

Install with:

```bash
pip install tensorflow numpy matplotlib pandas seaborn scikit-learn
```

## How to run

1. Open the notebook in Google Colab (recommended — free GPU) or a local Jupyter environment with internet access.
2. If using Colab: **Runtime → Change runtime type → GPU**.
3. Run all cells top to bottom. The dataset and pre-trained weights download automatically on first run.
4. Training takes roughly 15–25 minutes total on a Colab GPU (CNN: ~15 epochs, MobileNetV2: 8 head-training epochs + 6 fine-tuning epochs).
5. Fill in the accuracy values and observations in the Conclusion section once training completes.

## Notes

- Requires internet access to download CIFAR-10 and the ImageNet-pretrained MobileNetV2 weights.
- Training time and parameter-count figures depend on the environment (GPU vs. CPU) and are not pre-filled — capture them during your own run.
