
# 🍎 Apple Leaf Disease Detection  

Detect diseases in apple leaves using deep learning (PyTorch + Streamlit).  

---

## 🚀 Project Overview  
An end-to-end solution that classifies apple-leaf diseases using image recognition.  
A pretrained PyTorch model is deployed in a Streamlit web app for real-time inference.  

---

## **Key Highlights:**  
- Uses a CNN/ResNet-based model for disease detection.  
- Simple Streamlit UI for image upload and prediction.  
- Shows predicted class and confidence scores in chart format.  
- Helpful for precision agriculture and early-stage plant disease monitoring.

---

## 🧰 Repository Structure

```bash

├── app.py                          # Streamlit front-end application
├── model_arch.py                   # Model architecture definition
├── best_model.pth                  # Trained model weights
├── requirements.txt                # Dependencies
├── Apple leaf disease detection.ipynb   # Training notebook
├── Apple Leaf Disease Detection.pptx    # Project presentation
└── __pycache__/                    # Auto-generated cache files

```
## ⚙️ Installation & Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/YourUsername/AppleLeaf_Disease_Detection.git
cd AppleLeaf_Disease_Detection
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows  
source .venv/bin/activate  # macOS/Linux

```
### 3️⃣ Install dependencies manually
```bash
pip install streamlit torch torchvision torchaudio pillow numpy pandas matplotlib
```

### 4️⃣ Run the application
```bash
streamlit run app.py
```

## 🖥️ How to Use

**Step-by-step guide for users**

1. Run the Streamlit app (`streamlit run app.py`)  
2. Open `http://localhost:8501` in your browser  
3. Upload an apple leaf image  
4. See:  
   - 🧠 Predicted disease name  
   - 📊 Confidence chart  
   - 🌿 Healthy/Unhealthy status  
