# Midterm Exam Project

Welcome to the Midterm Exam Project for **CSST106-CS4B**! This project demonstrates advanced image recognition techniques using **HOG-SVM** and **YOLO** models.

---

## 📹 Project Overview Video
Click below to watch a detailed walkthrough of the project:
- [🎬 Project Video - 4B-VICTORIA-LAGANZON-MP](https://drive.google.com/file/d/1vBCyr6dTTU2CiiMqbl9YUUXW2ZzHPPt_/view?usp=sharing)

---

## 📑 Project Report
Dive into the detailed analysis and insights:
- [📄 Midterm Project Report (PDF)](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/4B-VICTORIA-LAGANZON-MP-report.pdf)

---

## 📂 Project Structure and Code

### 🚀 HOG-SVM Model
The HOG-SVM model is used for object detection by extracting Histogram of Oriented Gradients features and classifying them using a Support Vector Machine.

- **Code Resources:**
  - [🔗 HOG-SVM Model Code](https://github.com/laganzonj/CSST106-CS4B/tree/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/HOG-SVM_model)
  - [🔗 HOG-SVM Test Code](https://github.com/laganzonj/CSST106-CS4B/tree/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/HOG-SVM_test)
  - [📘 Jupyter Notebook](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/4B_VICTORIA_LAGANZON_MP_HOG_SVM.ipynb)
  - [🐍 Python Script](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/4b_victoria_laganzon_mp_hog_svm.py)

### ⚡ YOLO Model
The YOLO (You Only Look Once) model is a real-time object detection system that uses deep learning for high accuracy and speed.

- **Code Resources:**
  - [🔗 YOLO Model Code](https://github.com/laganzonj/CSST106-CS4B/tree/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/yolov5_model)
  - [🔗 YOLO Test Code](https://github.com/laganzonj/CSST106-CS4B/tree/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/yolov5_test)
  - [📘 Jupyter Notebook](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/4B_VICTORIA_LAGANZON_MP_YOLO.ipynb)
  - [🐍 Python Script](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/4b_victoria_laganzon_mp_yolo.py)

---

## ⚙️ Additional Resources

To facilitate testing and model configuration, the following resources are provided:

- [🗂 Data Configuration File](https://github.com/laganzonj/CSST106-CS4B/blob/4e871dcaaa2285a47274f05c91fa919b4e8bc978/4B-VICTORIA-LAGANZON-MP/temp_data.yaml)

---

## 🔍 About the Models

### HOG-SVM
The **Histogram of Oriented Gradients** (HOG) with **Support Vector Machine** (SVM) is used for object detection. HOG is particularly effective in capturing the structure and appearance of objects. By combining it with SVM, a robust classification of objects is achieved.

**Evaluation**:  
HOG-SVM proved effective in scenarios with simpler, more structured environments. Its reliance on handcrafted features, like edge orientation gradients, allowed it to classify specific, well-defined objects with reasonable accuracy while requiring minimal computational resources. However, it faced limitations with multi-scale detection, complex backgrounds, and generalization, which reduced its robustness in real-world settings with diverse object appearances and complex scenes.

### YOLO
**You Only Look Once** (YOLO) is an advanced deep-learning model for real-time object detection. Its speed and accuracy make it ideal for applications requiring immediate detection.

**Evaluation**:  
YOLOv5 showcased its strengths in real-time, multi-object detection across various environments. Leveraging deep learning and a one-shot detection framework, YOLOv5 achieved high accuracy and robustness, performing well in detecting multiple object classes even within cluttered and challenging backgrounds. While the model required extensive computational power, increasing training time, its versatility and efficiency in real-world applications, like surveillance and autonomous navigation, were evident.

---

