
# 👋 SignEmote: Real-Time Sign Language and Emotion Recognition with Text-to-Speech  

An **AI-powered assistive communication system** designed to recognize **sign language gestures** and **facial emotions** in real-time, converting them into **spoken language** using Google Text-to-Speech (gTTS).  
SignEmote aims to empower individuals with speech and hearing impairments by bridging the communication gap using deep learning and computer vision.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Demo](#demo)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

SignEmote is a unified real-time system that:
- Detects **hand gestures** representing sign language using **MediaPipe**
- Recognizes **facial emotions** using a CNN model trained on the **FER2013** dataset
- Converts the detected output into **speech** using **Google TTS API**

It’s designed to help **deaf, mute, or visually impaired individuals** by providing a seamless conversion from visual input to audio output.

---

## 🚀 Features

✅ Real-Time **Sign Language Detection**  
✅ Real-Time **Facial Emotion Recognition**  
✅ Converts Detected Text to **Speech** using Google TTS  
✅ **Highly Modular** – use Emotion/Sign/TTS independently or together  
✅ Easy-to-use CLI-based modes  
✅ Supports **7 emotions** and customizable hand gestures

---

## 🧠 Architecture

```
+--------------------+        +---------------------+       +-----------------+
|   Webcam Feed      | ---->  |  Sign & Emotion     | --->  |     Text        |
|                    |        | Detection (MediaPipe|       |  Generation     |
|                    |        | + CNN Model)        |       +-----------------+
+--------------------+                                        |
                                                              v
                                                      +---------------+
                                                      | Google TTS API|
                                                      +---------------+
                                                              |
                                                              v
                                                   🔊 Audio Output (Speech)
```

---

## 🗂 Dataset

### Emotion Detection
- **FER2013 Dataset**
  - 35,887 grayscale images (48x48)
  - 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised

### Sign Language Detection
- **Custom dataset** created using webcam
- Images labeled per gesture (A, B, C... or custom signs)

---

## 🛠 Getting Started

### 🔧 Requirements
- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- TensorFlow / Keras
- gTTS (Google Text-to-Speech)

### 🔽 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🏁 Running the Project

#### 1. 📷 Sign Language Detection
```bash
python collect_imgs.py               # Collect training images
python create_dataset.py             # Generate dataset
python train_classifier.py           # Train classifier
python inference_classifier.py       # Detect sign in real-time
```

#### 2. 😃 Emotion Detection
```bash
python train_emotion_model.py        # Train CNN model on FER2013
python detect_emotion.py             # Real-time emotion detection
```

#### 3. 🗣 Text-to-Speech (Auto-conversion)
The detected gesture or emotion is converted to audio using Google TTS.
```python
from gtts import gTTS
import os

text = "Hello, how are you?"
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")
os.system("start output.mp3")  # or use playsound module
```

---

## 📊 Results

| Module               | Accuracy        |
|----------------------|-----------------|
| Sign Language        | ~95% (on known signs) |
| Emotion Detection    | ~70–75% (on validation set) |
| TTS                  | Near real-time response |

---

## 🔭 Future Scope
- Support for continuous gesture-to-text conversion
- Multilingual speech output
- Mobile app deployment
- Expanded sign language dataset with dynamic signs

---

## 🤝 Contributing

We welcome all contributions!

1. Fork the repository  
2. Create a branch (`feature/my-feature`)  
3. Commit your changes  
4. Open a Pull Request 🚀

---

## 🛡 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.

---

## 💡 Acknowledgements

- [FER2013 Dataset – Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
- [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/)  
- [MediaPipe by Google](https://mediapipe.dev/)  
- [OpenCV](https://opencv.org/)  
- [gTTS – Google Text-to-Speech](https://pypi.org/project/gTTS/)
