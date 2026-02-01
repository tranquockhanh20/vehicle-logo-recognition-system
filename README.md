# Vehicle Logo Recognition using CNN
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/cc8e867b-eae2-403f-8de9-1e223cca048a" />

A lightweight deep learning project that applies Convolutional Neural Networks (CNNs) to recognize vehicle brand logos from images and real-time webcam input, designed for academic study and practical demonstration.

This project implements a **vehicle logo recognition system** based on **Convolutional Neural Networks (CNNs)**. The system is able to recognize car brand logos (VinFast, Ferrari and Toyota) from images and real-time webcam input.

---

## Overview

This project demonstrates how a CNN-based image classification model can be trained and deployed in a real-time application for vehicle logo recognition.

* Task: Vehicle logo classification
* Input: RGB image (224 × 224 × 3)
* Output classes:

  * Dang cho nhan dang ... (background / no logo)
  * VinFast
  * Ferrari
  * Toyota

The project includes a training pipeline and a real-time GUI application for inference.

---

## How it Works (Pipeline)

* Capture image from dataset (training) or webcam (inference)
* Resize and normalize the image to match model input format
* Feed the processed image into the CNN model
* Predict logo class and display result with confidence score

---

## Technologies Used

* **Python**
* **PyTorch** – model design and training
* **TensorFlow / Keras** – model loading and inference
* **OpenCV** – webcam capture and image processing
* **Tkinter** – desktop GUI
* **NumPy, Pillow, HDF5 (.h5)** – data handling and model storage

---

## Dataset Structure

```
preprocessed_data/
├── Dang cho nhan dang ...
├── Vinfast
├── Ferrari
└── Toyota
```

Each folder represents one class and follows the ImageFolder format.

---

## Model Architecture

* Conv2D (3 → 32) + ReLU + MaxPooling
* Conv2D (32 → 64) + ReLU + MaxPooling
* Flatten
* Fully Connected (128 neurons, ReLU)
* Output layer (3 classes)

Training details:

* Loss function: CrossEntropyLoss
* Optimizer: Adam

---

## Image Preprocessing

* Resize to 224 × 224
* Convert to tensor / NumPy array
* Normalize pixel values to range [-1, 1]

The same preprocessing is applied during training and inference.

---

## Training

Run training with:

```bash
python train.py
```

The trained model is saved in `.h5` format.

---

## Real-Time GUI Application

* Captures frames from a webcam
* Performs preprocessing and model prediction
* Displays predicted logo and confidence score
* Uses confidence threshold to avoid false recognition

---

## Conclusion

This project demonstrates a complete pipeline for **vehicle logo recognition using CNNs**, from data preparation and model training to real-time deployment with a graphical user interface. It serves as a practical application of neural networks in computer vision.
