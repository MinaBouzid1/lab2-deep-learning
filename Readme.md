# Lab 2 – Deep Learning & Applications: Computer Vision with PyTorch

**Dataset:** MNIST (60,000 training / 10,000 test grayscale images of handwritten digits, 28×28)
**Framework:** PyTorch / torchvision
**Models:** custom CNN, Faster R-CNN, fine-tuned VGG16 and AlexNet, Vision Transformer (ViT) built from scratch

---

## 1. Objective

Get familiar with the PyTorch library and build several neural architectures for computer vision, then compare them with the same metrics (accuracy, macro F1-score, loss, training time, number of parameters).

## 2. Repository content

| File | Description |
|---|---|
| `CNN_Classifier.ipynb` | Part 1: custom CNN, Faster R-CNN, VGG16 / AlexNet fine-tuning, comparison |
| `vist.ipynb` | Part 2: Vision Transformer from scratch |
| `README.md` | This report |

## 3. How to run

1. Open the notebooks in Google Colab or Kaggle.
2. Enable a GPU: `Runtime → Change runtime type → GPU`.
3. Run all cells from top to bottom. MNIST is downloaded automatically by `torchvision` (no manual download needed).
4. Experiment sizes (epochs, subset sizes) are set in the `CFG` dictionary at the top of `CNN_Classifier.ipynb`.

---

## 4. Part 1 – CNN, Faster R-CNN and pretrained models

### 4.1 Custom CNN
| Layer | Hyper-parameters | Output |
|---|---|---|
| Conv + BatchNorm + ReLU + MaxPool | 32 kernels 3×3, stride 1, padding 1, pool 2×2 | 32×14×14 |
| Conv + BatchNorm + ReLU + MaxPool | 64 kernels 3×3, stride 1, padding 1, pool 2×2 | 64×7×7 |
| Conv + BatchNorm + ReLU + MaxPool | 128 kernels 3×3, stride 1, padding 1, pool 2×2 | 128×3×3 |
| FC + ReLU + Dropout (0.5) | 1152 → 256 | 256 |
| FC | 256 → 10 | 10 |

Training: Adam (lr 1e-3), L2 regularisation (`weight_decay=1e-4`), Dropout, BatchNorm, StepLR scheduler, 5 epochs, batch size 128, full training set, GPU mode.

### 4.2 Faster R-CNN
Faster R-CNN is an object detector (Region Proposal Network + RoI head), so MNIST was converted into a detection problem:
- the **bounding box** is the tight box around the non-zero pixels of the digit,
- the **label** is the digit class + 1 (0 is reserved for the background),
- images are resized to 128×128 and replicated to 3 channels.

The `fasterrcnn_mobilenet_v3_large_fpn` backbone was used to keep training time reasonable (SGD, momentum 0.9, 3 epochs, subset of 6,000 training images). At inference, the class of the highest-scoring box is the predicted digit; an image with no detection counts as an error.

### 4.3 Fine-tuning VGG16 and AlexNet
Both models use ImageNet weights. The last fully connected layer is replaced by a 10-class layer, images are resized to 64×64, converted to 3 channels and normalised with ImageNet statistics. The convolutional layers are frozen (feature extraction) and the classifier is trained for 3 epochs on a subset of 12,000 training images.

## 5. Part 2 – Vision Transformer (from scratch)

Implementation based on the "Vision Transformers from Scratch (PyTorch)" guide:
1. **Patchify:** each 28×28 image is split into 7×7 = 49 patches of 4×4 pixels, each flattened to a vector of size 16.
2. **Linear embedding:** each patch is mapped to a vector of size `hidden_d = 8`.
3. **Class token:** a learnable token is prepended to the sequence of patches.
4. **Positional encoding:** fixed sinusoidal embeddings are added to keep the position of each patch.
5. **Encoder blocks (× 2):** LayerNorm → multi-head self-attention (2 heads) → residual → LayerNorm → MLP → residual.
6. **Classification head:** the final class token goes through a linear layer to give 10 logits.

Training: Adam (lr 0.005), cross-entropy loss, 5 epochs, batch size 128, about 1.6 K parameters. The notebook was run on CPU.

---

## 6. Results (MNIST test set)

| Model | Accuracy | F1 (macro) | Loss | Train time (s) | Params (M) |
|---|---|---|---|---|---|
| **CNN (custom)** | **0.9949** | **0.9949** | **0.0186** | 85.1 | 0.39 |
| Faster R-CNN | 0.9600 | 0.9595 | 0.5326* | 215.0 | 18.98 |
| VGG16 (fine-tuned) | 0.9617 | 0.9615 | 0.1240 | 54.1 | 134.30 |
| AlexNet (fine-tuned) | 0.9391 | 0.9382 | 0.1893 | 32.3 | 57.04 |
| ViT (from scratch) | 0.8614 | not computed | 0.4222 | not measured** | ≈0.0016 |

\* Detection loss (RPN + box regression + classification); it is not comparable to the cross-entropy of the other models.
\** The ViT notebook was run on CPU, so its time is not comparable with the GPU runs of Part 1.

## 7. Interpretation

### Part 1
- **Custom CNN – best overall.** Highest accuracy and F1 (99.49 %) and lowest loss, with only 0.39 M parameters. Convolution, pooling and weight sharing exploit the local structure of images, and BatchNorm, Dropout and L2 regularisation keep the gap between train and test loss small (no overfitting).
- **Faster R-CNN.** 96 % accuracy but the slowest model (215 s, 2.5× the CNN) and 49× more parameters. With a single centred digit per image, the RPN and box regression bring no benefit, and any missed detection is counted as an error.
- **VGG16 (fine-tuned).** 96.2 % with 134 M parameters. Its test loss was still decreasing at epoch 3, so more epochs or full fine-tuning would likely help.
- **AlexNet (fine-tuned).** Fastest (32 s) but least accurate of Part 1 (93.9 %) because of its smaller capacity; it also converged more slowly.
- **Why the pretrained models are behind the CNN.** ImageNet features come from colour natural images, while MNIST is low-resolution grayscale that has to be upscaled and replicated to 3 channels. They were also trained on a subset for only 3 epochs, while the CNN used the full dataset.

### Part 2 – ViT
- The ViT reaches **86.14 %** test accuracy (test loss 0.4222). The training loss decreases steadily (1.62 → 0.47) and the test loss is close to it, so the model is **under-fitted, not over-fitted**.
- **Causes:** (1) it is very small (hidden size 8, 2 heads, 2 blocks), so its capacity is limited; (2) only 5 epochs, the loss was still decreasing; (3) a ViT has no built-in locality or translation equivariance, so it must learn spatial structure from the data.
- **Possible improvements:** larger embedding (e.g. 64), more heads and blocks, more epochs, a learning-rate schedule, data augmentation, or pre-training.

### Comparison of ViT with Part 1
- The ViT is the **least accurate model** of the lab: about 13 points below the custom CNN, 10 below VGG16 and Faster R-CNN, and 8 below AlexNet.
- It is also **by far the smallest** (≈1.6 K parameters versus 0.39 M for the CNN and 134 M for VGG16), so its accuracy is high for its size.
- On MNIST, the CNN's image priors give the best accuracy / size / training-time trade-off.

## 8. Limitations
- The Faster R-CNN and the pretrained models were trained on subsets and for few epochs (3), while the CNN used the full dataset, so the comparison is not perfectly like-for-like.
- The ViT was run on CPU with a very small configuration and an F1-score was not computed for it.
- Each model was trained once (single seed), so small differences (e.g. VGG16 vs Faster R-CNN) are not statistically established.

## 9. Conclusion
For a small, simple, grayscale dataset like MNIST, a **compact CNN trained from scratch** is the best choice (99.49 % accuracy, 0.39 M parameters). Pretrained models and detectors are justified on more complex data (natural images, small datasets, multi-object scenes). ViTs need more capacity, more data or large-scale pre-training to reach their potential.

## 10. What I learned
- How to build, train and evaluate CNNs in PyTorch on GPU: convolution, padding, stride, pooling, dropout, L2 regularisation, optimizers and learning-rate scheduling.
- That Faster R-CNN is designed for object localisation and is not the right tool for plain classification, and how to convert a classification dataset into a detection dataset.
- How transfer learning works (replace the head, freeze or fine-tune layers, adapt input size and channels) and what it costs in parameters and memory.
- How a Vision Transformer works from scratch: patch embedding, class token, positional encoding, multi-head self-attention and encoder blocks.
- That the best model depends on the data and constraints (accuracy, size, training time), and that comparisons must use the same metrics, data and hardware to be fair.

## 11. References
- Dosovitskiy et al., *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020.
- Brian Pulfer, *Vision Transformers from Scratch (PyTorch): A Step-by-Step Guide*, Medium.
- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, 2015.
- Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG), 2014.
- Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet), 2012.
- PyTorch and torchvision documentation.