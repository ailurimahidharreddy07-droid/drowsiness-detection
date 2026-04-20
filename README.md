# Driver Drowsiness Detection (CNN + OpenCV)

This project implements a real-time driver drowsiness detection system using computer vision and deep learning. A Convolutional Neural Network (CNN) is trained to classify eye states as **Open** or **Closed**, and an alert is triggered when prolonged eye closure is detected.

---

## 🚀 Features

* Real-time webcam-based detection
* Eye state classification: **Open vs Closed**
* CNN model trained on grayscale eye images (64×64)
* Haar Cascade-based face and eye detection
* Alarm system triggered on drowsiness
* Lightweight and runs on CPU (no GPU required)

---

## 📂 Dataset

* Source: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
* Classes used:

  * `Closed`
  * `Open`

> Note: Other classes such as `yawn` and `no_yawn` were not used to maintain consistency with the inference pipeline.

---

## ⚙️ Preprocessing Pipeline

The same preprocessing is applied during both training and inference:

* Convert image to grayscale
* Resize to **64 × 64**
* Normalize pixel values to **[0, 1]**
* Reshape to **(64, 64, 1)**

---

## 🧠 Model Architecture

* 3 Convolutional layers (32 → 64 → 128 filters)
* MaxPooling after each convolution block
* Flatten layer
* Dense layer (256 units) with Dropout (0.5)
* Output layer: **2 neurons (Softmax)**

---

## 📊 Performance

* Accuracy: **~98%**
* Balanced Precision and Recall for both classes
* No significant overfitting observed

---

## 🖥️ File Structure

```
drowsiness-detection/
├── detect.py
├── Driver_Drowsiness_Detection.ipynb
├── drowsiness_model.h5
├── data/
│   ├── alarm.mp3
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_lefteye_2splits.xml
│   ├── haarcascade_righteye_2splits.xml
├── README.md
```

---


1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure required files are present:

* `drowsiness_model.h5`
* Haar cascade XML files (inside `data/`)
* `alarm.mp3`

4. Run the detection script:

```bash
python detect.py
```

Press **`q`** to exit.

---
## 🔄 How It Works

1. Webcam captures real-time frames
2. Haar cascades detect face and eyes
3. Eye regions are preprocessed
4. CNN predicts eye state (Open / Closed)
5. If both eyes remain closed for **≥10 frames**, an alarm is triggered

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Haar cascades may fail under extreme angles
* Performance may drop with occlusions (e.g., sunglasses)

---


