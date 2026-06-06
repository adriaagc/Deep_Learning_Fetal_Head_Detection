# Fetal Head Detection in Ultrasound Images using MobileNetV3 Regression

This repository contains the complete PyTorch implementation and experimental framework for automating the measurement of fetal head circumference from ultrasound images using deep learning regression models. This project was developed as a Deep Learning Final Project by Adrià García, Manel Gutierrez & Ernest Fosch.

Monitoring fetal growth stands out as a key indicator of healthy prenatal care. By utilizing a lightweight backbone like MobileNetV3, this project maps ultrasound frames directly to localized mathematical parameters of the fetal skull ellipse, minimizing human sonographer error and making real-time clinical applications viable.

**Methodology Overview**

Instead of traditional pixel-wise segmentation models (e.g., standard U-Net architectures), this approach frames fetal head boundary localization as a direct parameter regression task.
1. Mathematical Ellipse Extraction

The ground truth binary masks are automatically parsed to extract an outer boundary contour using OpenCV. This contour is then fitted to an analytical ellipse defined by a normalized 6-dimensional parameter vector $\mathbf{y}$:

$$\mathbf{y} = [X_c, Y_c, d_{max}, d_{min}, \sin(2\theta), \cos(2\theta)]$$
