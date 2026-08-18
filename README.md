# 🔬 MicroPy-Align: CLEM Image Registration & Coordinate Mapping Engine

An open-source Python tool designed for Correlative Light and Electron Microscopy (CLEM) workflows, providing automated image alignment (SIFT/ORB), fiducial-based manual affine transformations, and cross-modal coordinate mapping with real-time UI.

---

## 🎬 Demo Preview

![Demo](micropy-align-demo.gif)

---

## 🌟 Key Features

- **Multi-Modal Image Registration**:
  - **Automatic Registration**: Employs SIFT/ORB feature detection and RANSAC for automatic affine matrix calculations.
  - **Manual Fiducial Alignment**: Fallback mechanism using matching control points for low-contrast/cross-modal images.
- **Precision Stage Navigation & Coordinate Mapping**:
  - Calculates $2 \times 3$ Affine Transformation Matrices to map $2D$ coordinates $(x, y) \to (x', y')$ from fluorescence targets to electron microscopy frames.
- **Interactive Dual-Channel Overlay**:
  - Real-time Streamlit Web UI with adjustable Alpha transparency slider for seamless fusion visualization.

---

## 🏗️ Project Architecture

```text
Fluorescence Image + EM Image ──► Streamlit Web UI (app.py)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       [ImageAligner (SIFT/ORB/Fiducials)]       [CoordinateMapper Engine]
                    │                                             │
                    ▼                                             ▼
          Affine Matrix (2x3) ──────────────────►  Cross-Modal Coordinates (x', y')
                    │
                    ▼
          Dual-Channel Fusion View (Alpha Overlay)

---

## 🛠️ Tech Stack
Computer Vision: OpenCV, NumPy, SciPy

Frontend / Interaction: Streamlit

Image Processing: Pillow, Matplotlib

---

## 🚀 Quick Start
1. Clone & Setup
git clone [https://github.com/Yooyoo24/micropy-align.git](https://github.com/Yooyoo24/micropy-align.git)
cd micropy-align
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run Application
streamlit run app.py

---

## 📝 License
Distributed under the MIT License.
