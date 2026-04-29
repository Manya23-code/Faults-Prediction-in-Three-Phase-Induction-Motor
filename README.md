# ⚡ Fault Prediction of Three-Phase Induction Motor
**A Deep Learning Approach using CNN-LSTM & Attention Mechanism**

## 📖 Project Overview
This project focuses on the real-time health monitoring and fault diagnosis of 3-Phase Induction Motors. By leveraging **Industrial IoT (IIoT)** and **Deep Learning**, we can predict critical electrical and mechanical faults before they lead to system failure.

The system processes **Magnetic Flux** and **Motor Speed** data to classify the motor's state into 5 categories with high precision.

## 🚀 Key Features
- **Hybrid AI Architecture:** Uses **CNN** for spatial feature extraction and **LSTM** for temporal sequence analysis.
- **Attention Mechanism:** A dedicated layer to focus on significant sensor fluctuations during fault events.
- **Edge Deployment:** Optimized model (55KB) deployed on **ESP32** using TensorFlow Lite Micro.
- **Real-time Monitoring:** Web dashboard built with **Streamlit** for live data visualization.

## 🛠️ Tech Stack
- **Languages:** Python (AI/ML), C++ (Hardware/ESP32).
- **Libraries:** TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, EloquentTinyML.
- **Signal Processing:** Fast Fourier Transform (FFT), Min-Max Normalization.
- **Hardware:** ESP32 Microcontroller, Flux Sensors, Speed Sensors.

## 📊 Faults Detected
1. **Healthy / Normal Load** (Stable operation)
2. **No-Load Condition** (Idling state)
3. **High Resistance** (Stator/Rotor issues)
4. **Overload Fault** ⚠️ (Mechanical stress)
5. **Single Phase Open** 🚨 (Critical Electrical Fault)

## 📂 Repository Structure
- `1.py`: Data pre-processing & FFT analysis.
- `2.py`: Model training & Attention layer implementation.
- `convert.py`: TFLite conversion & quantization.
- `make_header.py`: Generates C++ header for ESP32.
- `temp.ino`: Final hardware deployment code.
- `model_data.h`: The AI brain in Hex format.

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Manya23-code/Faults-Prediction-in-Three-Phase-Induction-Motor.git](https://github.com/Manya23-code/Faults-Prediction-in-Three-Phase-Induction-Motor.git)