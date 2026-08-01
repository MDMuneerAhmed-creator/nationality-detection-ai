# Nationality & Feature Detection AI (Trained Completely From Scratch)

A full-stack Deep Learning AI application built completely from scratch using **4 custom Convolutional Neural Networks (CNNs)** with **zero pretrained models, weights, transfer learning, or cloud AI APIs**.

---

## 📌 Strict Architectural Constraints
- ❌ NO ResNet, MobileNet, EfficientNet, VGG, DenseNet, YOLO, CLIP, FaceNet, ArcFace, or MTCNN
- ❌ NO Transfer Learning or Pretrained Checkpoints
- ❌ NO OpenAI, Gemini, Hugging Face, or Cloud AI APIs
- ✅ 100% Custom CNN Models created layer-by-layer using `Conv2D`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `Dropout`, `Linear` (Dense), and `Softmax`
- ✅ Every model is initialized with random weights (He/Kaiming Normal) and trained strictly on local project datasets

---

## 🎯 Model Architecture & Objectives

The application loads **4 independent Scratch CNN models**:

1. **Model 1: Nationality Classification CNN** (`Nationality_CNN`) -> Predicts: Indian, American, African, Japanese, Chinese, German, French
2. **Model 2: Emotion Classification CNN** (`Emotion_CNN`) -> Predicts: Happy, Sad, Angry, Fear, Neutral, Disgust, Surprise
3. **Model 3: Age Classification CNN** (`Age_CNN`) -> Predicts: 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 60+
4. **Model 4: Dress Colour Classification CNN** (`Dress_Colour_CNN`) -> Predicts: Black, Blue, White, Green, Yellow, Red, Orange, Pink, Purple, Brown, Grey

---

## 🔀 Conditional Output Logic

When an image is uploaded, Model 1 predicts **Nationality** and Model 2 predicts **Emotion**. Then, conditional rules determine whether Model 3 (Age) and Model 4 (Dress Colour) are executed:

- **Indian**: Predicts **Nationality**, **Emotion**, **Age**, AND **Dress Colour**
- **American**: Predicts **Nationality**, **Emotion**, AND **Age**
- **African**: Predicts **Nationality**, **Emotion**, AND **Dress Colour**
- **All Other Nationalities** (Japanese, Chinese, German, French): Predicts **Nationality** and **Emotion** ONLY

---

## 📂 Project Directory Structure

```
project/
├── backend/
│   ├── main.py               # FastAPI server endpoints (/predict, /upload, /health)
│   ├── database.py           # SQLite database logger (predictions, metrics, training history)
│   ├── models.py             # Scratch CNN architectures (PyTorch / Layer definitions)
│   ├── pipeline.py           # Dataset pipeline (128x128 resize, RGB, normalization, augmentations)
│   └── evaluator.py          # Metrics calculator (Accuracy, Precision, Recall, F1, Confusion Matrix)
├── training/
│   └── train_all_models.py   # Training script with EarlyStopping, ReduceLROnPlateau, Checkpointing
├── utils/
│   └── dataset_generator.py  # Dataset directory setup & synthetic generator
├── datasets/
│   ├── nationality/          # Indian/, American/, African/, Japanese/, etc.
│   ├── emotion/              # Happy/, Sad/, Angry/, Fear/, etc.
│   ├── age/                  # 0-10/, 11-20/, 21-30/, etc.
│   └── dress_color/          # Black/, Blue/, White/, Red/, etc.
├── saved_models/             # Saved .pt model checkpoints
├── notebooks/
│   └── model_experiments.ipynb # Training experiments notebook
├── src/                      # React 19 + TypeScript + Tailwind GUI Frontend
├── server.ts                 # Full-stack Node.js server bridge
├── README.md
└── requirements.txt
```

---

## 🚀 Installation & Running Guide

### 1. Backend Setup & Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Generate Dataset & Train All 4 Models From Scratch
```bash
python3 training/train_all_models.py
```

### 3. Run Application
```bash
npm run dev
```

---

## 🧪 Testing Guide

1. Open the web interface at `http://localhost:3000`.
2. Select a preset sample image or upload a custom facial image (`.jpg`, `.png`).
3. Click **"Predict with Custom CNNs"**.
4. Observe the live stage-by-stage forward pass animation through the scratch CNNs.
5. Check the output card displaying **Nationality**, **Emotion**, and conditional **Age** or **Dress Colour** badges.
6. Explore the **Model Inspector** tab to view layer-by-layer specs and zero-pretrained parameter counts.
7. Open the **Training Dashboard** tab to view loss/accuracy curves and confusion matrices.
