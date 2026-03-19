# 🐙 Sentiment Analysis of Cephalopods – GSoC 2026 Entry Task

## 📌 Overview
This repository contains my submission for the GSoC 2026 entry task for the project **"Sentiment Analysis of Cephalopods"**.

The goal of this task is to demonstrate:
- Basic video data handling
- Simple and interpretable feature extraction
- Visualization of behavioral signals
- Reasoning about possible behavioral states

This implementation focuses on lightweight, interpretable methods rather than complex models, in line with the exploratory nature of the task.

---

## 🎥 Dataset
A short aquatic animal video (10–30 seconds) is used.

- File: `data/octopus_camouflage.mp4`
- If not included due to size, place your own video in the `data/` folder.

---

## ⚙️ Features Implemented

### 1. Motion Magnitude
- Computed using frame differencing
- Measures how much the scene changes between frames
- Indicates activity level

### 2. Histogram Change
- Computed using grayscale histogram differences
- Captures changes in intensity distribution
- Reflects motion, posture, or appearance variations

---

## 📊 Video Information Extracted

The script reports:
- Frame rate (FPS)
- Resolution
- Total frames
- Duration

---

## 📈 Visualization

The following outputs are generated:

- **Feature Plot**
  - Motion vs Time
  - Histogram Change vs Time
- **Sample Frames**
  - Beginning, middle, end frames

These help identify:
- High activity periods
- Low activity periods

---

## 📂 Project Structure

```bash
cephalopod-entry-task/
│── data/
│ └── sample_video.mp4
│
│── notebooks/
│ ├── analysis.ipynb
│
│── outputs/
│ ├── feature_plot.png
│ └── sample_frames.png
│
│── src/
│ └── extract_features.py
│
│── requirements.txt
│── README.md
```
---

## 🚀 Setup Instructions

### 1. Clone repository

```bash
git clone [https://github.com/melonmusk20/cephalopod_sentiment_analysis.git]
cd cephalopod sentiment analysis
```

### 2. Install dependencies

```bash 
pip install -r requirements.txt
```

### 3. Run the script 

```bash
python src/extract_features.py
```

---

## 📤 Outputs

Generated files (in outputs/):

• feature_plot.png

• sample_frames.png

---

## 🧠 Behavioral Analysis

The motion magnitude feature provides a simple estimate of activity over time. Higher values may indicate rapid movement, escape behavior, or active exploration, while lower values may correspond to resting or calm states.

The histogram change feature captures variations in overall visual appearance, which may reflect movement, posture changes, or body orientation.

When both features increase together, it may suggest heightened activity or behavioral transitions. When both remain low, the animal may be relatively still.

However, these features are limited because they do not directly capture posture, shape, or fine-grained behavioral cues. They are also sensitive to camera motion, lighting changes, and background disturbances.

Additional modalities such as:

 • posture tracking

 • contour analysis

 • audio signals

could significantly improve behavioral inference.

---

## ⚠️ Limitations

   • No direct posture or body-part tracking

   • Sensitive to lighting and camera movement

   • Cannot infer true emotional states

   • Works as a baseline, not a full behavioral model

##  Future Improvements
 
   • Optical flow-based motion analysis

   • Contour and shape tracking

   • Multi-modal integration (audio + video)

   • Deep learning models for behavior classification
