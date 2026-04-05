# Emotion Recognition using Hybrid Fuzzy–GA–NN Approach

## 📌 Project Overview

This project implements an **Emotion Recognition System** using a hybrid approach:

* **Neural Network (CNN)** for emotion classification
* **Fuzzy Logic** to handle uncertainty in prediction confidence
* **Genetic Algorithm (GA)** (extendable) for optimization

The system detects human emotions (Happy, Sad, Angry, etc.) from facial expressions using a webcam.

---

## 🚀 Features

* Real-time emotion detection using webcam
* CNN-based facial emotion classification
* Fuzzy logic for emotion intensity interpretation
* Modular and extensible architecture
* Clean and beginner-friendly implementation

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* scikit-fuzzy
* DEAP (Genetic Algorithm)

---

## 📁 Project Structure

emotion-hybrid/
│── dataset/
│ ├── train/
│ ├── test/
│── model/
│── train_model.py
│── app.py
│── fuzzy_logic.py
│── ga_optimize.py
│── README.md

---

## ⚙️ Prerequisites (IMPORTANT)

Before running the project, make sure you have:

* Python 3.8 or above installed
* pip (Python package manager)
* Webcam (for real-time detection)

---

## 📦 Step 1: Install Dependencies

Run the following command:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-fuzzy deap
```

---

## 📂 Step 2: Setup Dataset

Ensure your dataset is in this format:

dataset/
├── train/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   └── ...
├── test/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   └── ...

---

## 🧠 Step 3: Train the Model

Run:

```bash
python train_model.py
```

This will:

* Train the CNN model
* Save the model in the `model/` folder

---

## 🎥 Step 4: Run the Application

After training is complete, run:

```bash
python app.py
```

This will:

* Open your webcam
* Detect faces
* Display emotion + fuzzy intensity

Press **ESC** to exit the application.

---

## ⚠️ Important Notes

* Do not upload dataset or model files to GitHub (use `.gitignore`)
* Training may take time depending on system performance
* Ensure good lighting for better detection accuracy

---

## 🔥 Future Improvements

* Add Flask/React dashboard
* Improve accuracy using transfer learning
* Add voice emotion recognition
* Integrate real GA-based optimization

---

## 💡 Author

Developed as part of an AI-based project on Emotion Recognition using Hybrid Techniques.

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
