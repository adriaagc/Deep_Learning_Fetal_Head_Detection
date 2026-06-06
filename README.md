# Fetal Head Detection in Ultrasound Images using MobileNetV3 Regression

This repository contains the complete PyTorch implementation and experimental framework for automating the measurement of fetal head circumference from ultrasound images using deep learning regression models. This project was developed as a Deep Learning Final Project by Adrià García, Manel Gutierrez & Ernest Fosch.

Monitoring fetal growth stands out as a key indicator of healthy prenatal care. By utilizing a lightweight backbone like MobileNetV3, this project maps ultrasound frames directly to localized mathematical parameters of the fetal skull ellipse, minimizing human sonographer error and making real-time clinical applications viable.

## **Methodology Overview**

Instead of traditional pixel-wise segmentation models (e.g., standard U-Net architectures), this approach frames fetal head boundary localization as a direct parameter regression task.

### 1. Mathematical Ellipse Extraction

The ground truth binary masks are automatically parsed to extract an outer boundary contour using OpenCV. This contour is then fitted to an analytical ellipse defined by a normalized 6-dimensional parameter vector $\mathbf{y}$:

$$\mathbf{y} = [X_c, Y_c, d_{max}, d_{min}, \sin(2\theta), \cos(2\theta)]$$

$X_c, Y_c$: Center spatial coordinates of the ellipse (normalized by image size).
$d_{max}, d_{min}$: Maximum (major) and minimum (minor) axes lengths (normalized by image size).
$\theta$: Rotation angle converted into sine and cosine components to ensure periodicity and continuous gradients during backpropagation.

### 2. Network Architectures

We evaluate the small and large version of MobileNet v3, both models using pre-trained ImageNet weights. Since MobileNetv3 is built to predict objects among 1000 classes we must custom the final classifier layer so that the network converts the final feature maps to the 6 continuous output targets.

## **Environment & Requirements**

The project pipeline runs on standard cloud GPU environments (e.g., Google Colab T4) or local workstations equipped with CUDA. Ensure you have Python 3.10+ and the following packages installed:
pip install torch torchvision numpy opencv-python matplotlib

## **Step-by-Step Reproduction Guide**

**Step 1: Data Preparation**

Download the HC18 Fetal Head Circumference dataset.
Store the zipped files inside the data/ folder exactly as shown in the file structure. The data loader handles zip reads without requiring extraction to disk.

**Step 2: Dataset Validation & Pipeline Setup**

The script executes a Train/Validation split of 80/20 on the 799 training samples, leaving a final data split of:
- Training Subset: 639 images
- Validation Subset: 160 images
- Test Dataset: 200 images
Images and masks are downscaled to a unified spatial resolution of $256 \times 256$ pixels.

**Step 3: Run Training & Evaluation**

Open the notebook MobileNetv3_Ellipse_Points.ipynb and execute the cells sequentially to instantiate the dataset and use OpenCV contour tracking to parse masks into normalized vectors, define data augmentation: RandomRotation (15 degrees), RandomAffine transforms (±10% translation, 0.9-1.1 scale zoom) with bilinear interpolation and ColorJitter to change the brightness and contrast. Then train the models using Mean Squared Error (MSE) loss on the normalized target parameters, utilizing a default batch size of 16 to try to stabilize variance gradients.

**Step 4: Layer-Freezing Experiments**

First, on the MobileNetv3_small depending on the amount of layers you freeze you will have different number of parameters and performances:
- Total parameters: 1,001,638
- Classifier Only (Linear Head): Freezes the whole backbone (74,630 trainable parameters).
- Last 2 Layers + Classifier: 425,174 trainable parameters.
- Last 4 Layers + Classifier: 811,118 trainable parameters.
- Last 6 Layers + Classifier: 862,886 trainable parameters.

Second, on the MobileNetv3_large:
- Total parameters: 3,095,734
- Last 6 layers + Classifier: 2,903,790 trainable parameters.

