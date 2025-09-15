# PDVideoAnalysis

**A Machine Learning Pipeline with Computer-Vision Methods for Quantifying Standardized Motor Assessment in Parkinson’s Disease**

This repository contains code and resources for using computer vision and machine learning to analyze standardized motor tasks in Parkinson’s disease (PD). The project was developed as a research project at Northwestern University and Shirley Ryan AbilityLab, and is summarized in the accompanying [poster](./poster.pdf).

---

## 🚀 Overview

Parkinson’s disease (PD) is a progressive neurodegenerative disorder that primarily affects motor function. Clinical motor assessments, such as the **MDS-UPDRS**, are time-consuming, infrequent, and require in-person visits.  

**Goal:**  
Develop a video-based pipeline to automatically predict standardized motor assessment scores for PD patients, assisting clinicians with more efficient and accessible monitoring.

---

## 📋 Methodology

The pipeline consists of:

1. **Video Processing** – Segment and preprocess raw video.
2. **Pose Estimation** – Extract 2D/3D body keypoints using mesh regression.
3. **Kinematic Calculation** – Compute joint angles, speeds, accelerations, and body swing measures.
4. **Post-Processing** – Temporal smoothing and noise reduction.
5. **Feature Engineering** – Derive clinically relevant features (e.g., tremor frequency, step frequency, elbow angular velocity).
6. **Label Encoding** – Convert clinical scores into ordinal categories.
7. **Supervised Learning** – Train machine learning models (e.g., Ordinal Random Forest) for prediction.

---

## 📊 Results

- **Tasks Evaluated:**  
  - Gait  
  - Finger-to-nose (left & right)  

- **Dataset:**  
  - 28 subjects  
  - 168 sessions (ON vs OFF medication states)  

- **Key Findings:**  
  - Overall score prediction accuracy: **61%**  
  - Bradykinesia score prediction accuracy: **60%**  
  - Strongest predictive features:  
    - **Gait:** Step frequency  
    - **Finger-to-nose:** Median elbow angular speed  

See the full poster for figures, results, and discussion:  
📄 [Poster (PDF)](./poster.pdf)

---

## 🔧 Installation

Clone this repository:

```bash
git clone https://github.com/caraido/PDVideoAnalysis.git
cd PDVideoAnalysis
