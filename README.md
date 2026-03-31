# 😷 Face Mask Detection System

A deep learning-based web application that detects whether a person is wearing a face mask or not using image input.

Built using **TensorFlow, CNN, and Streamlit**, this project demonstrates real-world computer vision deployment.

---

## 📌 Features

* 📷 Upload image for detection
* 🤖 CNN-based classification model
* 📊 Displays prediction with confidence score
* 🎨 Clean and interactive Streamlit UI
* ⚡ Fast and lightweight

---

## 🧠 Model Details

* Architecture: Convolutional Neural Network (CNN)
* Input Size: 128 × 128 × 3
* Classes:

  * 0 → Without Mask ❌
  * 1 → With Mask ✅
* Accuracy: ~92% on test data

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pillow
* Streamlit

---

## 📂 Project Structure

```
Face Mask Detection/
│── app.py
│── face_mask_model.pkl / .h5
│── requirements.txt
│── test_images/
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/HARSH-GOHIL-git/face-mask-detection.git
cd face-mask-detection
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the application

```
streamlit run app.py
```

---

## 📸 Usage

1. Upload an image (jpg/png)
2. Model processes the image
3. Get prediction:

   * ✅ Mask
   * ❌ No Mask
4. View confidence score

---

## 📊 Sample Output

```
✅ Safe: Person is wearing a Mask  
Confidence: 94.32%
```

---

## ⚠️ Note

* Make sure the model file (`.pkl` or `.h5`) is present in the root directory
* If not included in repo, download it separately and place it in the project folder

---

## 🚀 Future Improvements

* 🎥 Real-time webcam detection
* 📦 Deployment on cloud (Streamlit / Render)
* 🔍 Face detection with bounding boxes
* 📱 Mobile-friendly UI

---

## 🤝 Contributing

Feel free to fork this repo and improve the project!

---

## 📧 Contact

**Harsh Gohil**

* GitHub: https://github.com/HARSH-GOHIL-git/face-mask-detection

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it really helps!