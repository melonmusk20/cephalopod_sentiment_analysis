# 🐙 Sentiment Analysis of Cephalopods

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
│ └── octopus_camouflage.mp4
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
git clone https://github.com/melonmusk20/cephalopod_sentiment_analysis.git
cd cephalopod_sentiment_analysis
```

### 2. Install dependencies

```bash 
pip install -r requirements.txt
```

### 3. Run the script 

```bash
python src/extract_features.py
```

## 4. Notebook instruction

```bash
   Open notebooks/analysis.ipynb to view the exploratory workflow and plots.
```
---

## 📊 Sample Outputs

### 📈 Feature Plot
<img src="outputs/feature_plot.png" width="700">

### 🎞️ Sample Frames
<img src="outputs/sample_frames.png" width="700">
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

## Key Observations

- Motion magnitude shows noticeable spikes around 3–5 seconds, indicating increased activity.
- Histogram change also rises near these intervals, suggesting simultaneous visual variation.
- Periods with low motion and low histogram change correspond to relatively stable or calm behavior.

---

## ⚠️ Limitations

- Motion magnitude is sensitive to camera movement and noise.
- Histogram change can be affected by lighting variations.
- These features do not capture posture or fine-grained body dynamics.
- Behavioral sentiment cannot be reliably inferred from video alone.

---

##  Future Work

- Optical flow for richer motion representation
- Contour-based analysis for body shape and posture
- Multi-modal integration (audio + video)
- Temporal modeling for behavior sequences
