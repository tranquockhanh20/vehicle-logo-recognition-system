# Vehicle Logo Recognition using Convolutional Neural Networks (CNN)

This project focuses on building a **deep learning–based vehicle logo recognition system** using **Convolutional Neural Networks (CNNs)**. The system is capable of recognizing car brand logos (e.g., **VinFast** and **Ferrari**) from images and live camera input.

The project includes:

* A **training pipeline** for building a CNN model.
* A **real-time GUI application** for logo recognition using a webcam.
* A complete **image preprocessing and inference pipeline**.

---

## 1. Project Objectives

* Design and train a CNN model to classify vehicle logos.
* Apply image preprocessing techniques to improve recognition accuracy.
* Deploy the trained model into a real-time desktop GUI application.
* Demonstrate practical application of neural networks in computer vision.

---

## 2. Technologies Used

### Programming Language

* **Python 3.x**

### Deep Learning Frameworks

* **PyTorch** – used for building and training the CNN model.
* **TensorFlow / Keras** – used for loading and running the trained model in the GUI.

### Computer Vision & Image Processing

* **OpenCV (cv2)** – webcam capture and image processing.
* **Pillow (PIL)** – image conversion for GUI display.

### GUI Development

* **Tkinter** – building a real-time desktop application interface.

### Data Handling

* **NumPy** – numerical operations and image array manipulation.
* **h5py / HDF5 (.h5)** – storing trained model weights.

---

## 3. Dataset Structure

The dataset is organized using the `ImageFolder` format:

```
preprocessed_data/
├── Dang cho nhan dang ...   # Background / no-logo images
├── Vinfast                 # VinFast logo images
└── Ferrari                 # Ferrari logo images
└── Toyota                  # Toyota logo images
```

Each folder represents one class. The model output indices correspond directly to these folder names.

---

## 4. Model Architecture

The CNN architecture is designed as follows:

1. **Input Layer**: RGB image of size `224 × 224 × 3`
2. **Convolution Layer 1**:

   * Conv2D (3 → 32 filters, kernel size 3×3)
   * ReLU activation
   * MaxPooling (2×2)
3. **Convolution Layer 2**:

   * Conv2D (32 → 64 filters, kernel size 3×3)
   * ReLU activation
   * MaxPooling (2×2)
4. **Flatten Layer**
5. **Fully Connected Layer**:

   * Dense (128 neurons, ReLU)
6. **Output Layer**:

   * Dense (3 neurons – background, VinFast, Ferrari)

The network is trained using:

* **Loss function**: CrossEntropyLoss
* **Optimizer**: Adam

---

## 5. Image Preprocessing Pipeline

During training and inference, images are processed as follows:

1. Resize image to `224 × 224`.
2. Convert image to tensor / NumPy array.
3. Normalize pixel values to the range `[-1, 1]` using:

```
normalized_pixel = (pixel / 127.5) - 1
```

This ensures consistency between training and real-time prediction.

---

## 6. Training Process

* Dataset is split into **80% training** and **20% testing**.
* Training is performed using PyTorch.
* Model parameters are saved in **HDF5 format (.h5)**.

To train the model:

```bash
python train.py
```

---

## 7. Real-Time GUI Application

The GUI application:

* Captures video frames from a webcam.
* Preprocesses each frame.
* Performs logo prediction using the trained model.
* Displays the predicted logo name and confidence score.

Key features:

* Real-time recognition
* Confidence threshold to avoid false predictions
* Safe error handling to prevent application freezing

---

## 8. Class Labels

The class labels are defined directly in the GUI code:

```python
class_names = [
    "Dang cho nhan dang ...",
    "Vinfast",
    "Ferrari",
    "Toyota"
]
```

This order must match the model output indices.

---

## 9. Results

* The system can correctly recognize VinFast and Ferrari logos under good lighting conditions.
* The background class helps prevent false recognition when no known logo is present.
* Real-time performance is stable with appropriate update intervals.

---

## 10. Limitations and Future Improvements

### Current Limitations

* Limited number of logo classes.
* Accuracy depends on image quality and lighting.
* CNN architecture is relatively simple.

### Future Improvements

* Use a pre-trained model (MobileNet, EfficientNet).
* Apply data augmentation.
* Convert model to **TensorFlow Lite** for faster inference.
* Add more vehicle brands.

---

## 11. Conclusion

This project demonstrates a complete pipeline for **vehicle logo recognition using neural networks**, from data preparation and model training to real-time deployment. It highlights the practical application of CNNs in computer vision and serves as a solid foundation for further research and development.

---

## Author

* **Student Project – Neural Networks / Computer Vision**
* Developed for academic purposes
