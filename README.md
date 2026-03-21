<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&amp;color=gradient&amp;customColorList=5,11,18&amp;height=180&amp;section=header&amp;text=AI%20Sensor%20Exercises&amp;fontSize=42&amp;fontColor=fff&amp;animation=twinkling&amp;fontAlignY=38" />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&amp;logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&amp;logo=scikitlearn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&amp;logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Practical AI and sensor data exercises: classification, anomaly detection, and real-time signal analysis.**

</div>

## Overview

A collection of hands-on exercises working with real sensor data streams. Each notebook covers one complete problem end-to-end: data ingestion, feature engineering, model training, and evaluation.

## Exercises

### Exercise 1: Sensor Anomaly Detection
Detect faulty sensor readings in a time-series stream using isolation forest and z-score methods. Compare detection rates and false positive rates across both approaches.

### Exercise 2: Activity Classification
Classify physical activities (walking, running, standing, cycling) from accelerometer and gyroscope readings using Random Forest and SVM classifiers.

### Exercise 3: Signal Preprocessing Pipeline
Build a reusable preprocessing pipeline for noisy sensor signals: low-pass filtering, normalization, window segmentation, and feature extraction.

### Exercise 4: Real-Time Inference
Deploy a trained classifier for real-time inference on a simulated data stream. Measure throughput and latency under different batch sizes.

## Skills Covered

- Time-series feature engineering (rolling stats, FFT features, peak detection)
- Supervised classification (Random Forest, SVM, k-NN)
- Unsupervised anomaly detection (Isolation Forest, DBSCAN)
- Model evaluation on imbalanced datasets (precision/recall, AUC-ROC)
- Signal processing basics (filtering, resampling, normalization)

## Quick Start

```bash
git clone https://github.com/Aliipou/Ai-Sensor-Exercise.git
cd Ai-Sensor-Exercise
pip install -r requirements.txt
jupyter notebook
```

## Requirements

```
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scipy>=1.11
jupyter>=1.0
```

## License

MIT
