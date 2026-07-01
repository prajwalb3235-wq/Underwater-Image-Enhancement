# Underwater Image Enhancement using Random Forest Regression

## Overview

This project implements an underwater image enhancement pipeline using Random Forest Regression trained on paired underwater and reference images.

Two feature extraction approaches are explored and compared:

- Pixel-only features
- 3×3 patch features with global image statistics

The quality of the enhanced images is evaluated using standard image quality metrics.

---

## Features

- Image preprocessing using OpenCV
- Random pixel sampling for training
- Data augmentation using horizontal and vertical flipping
- Random Forest Regression for pixel-wise enhancement
- Patch-based feature extraction
- Global statistical feature extraction
- Image reconstruction
- Performance evaluation using:
  - PSNR
  - SSIM
  - Mean Squared Error (MSE)
- Visualization of best and worst performing predictions

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- scikit-image

---

## Dataset

This project was developed using a paired underwater image dataset provided by the project mentor for academic purposes.

For privacy and licensing reasons, the dataset is **not included** in this repository.

The code expects the following directory structure:

```
Train_Val_Dataset/
    train/
        raw/
        reference/
    val/
        raw/
        reference/
```

## My Contribution

Implemented the Random Forest Regression model and feature engineering pipeline as part of a team project comparing multiple machine learning approaches for underwater image enhancement.

---

## Future Improvements

- CNN-based enhancement models
- U-Net implementation
- GPU acceleration
- Hyperparameter optimization
- Model serialization and deployment

Note: This repository contains my implementation of the Random Forest Regression approach developed as part of a team project. Other team members explored alternative machine learning models independently.
