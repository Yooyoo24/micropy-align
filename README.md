# MicroPy-Align: CLEM Image Registration & Coordinate Mapping Engine

An open-source, modular Python framework designed for Correlative Light and Electron Microscopy (CLEM) workflows. It provides automated image registration, 2D physical coordinate transformation, and real-time dual-channel fusion visualization.

---

## Motivation

In Correlative Light and Electron Microscopy (CLEM), researchers identify functional points of interest (POIs) under a fluorescence microscope and navigate to the exact nanoscale structures under an electron microscope (EM). Due to rotational, scale, and translational discrepancies between the two imaging modalities, manual alignment is error-prone. **MicroPy-Align** automates cross-modal image registration and stage coordinate mapping.

---

## Demo Preview

![Demo](micropy-align-demo.gif)

---

## CLEM Workflow

```text
[ Fluorescence Image ] ──┐
                         ├──► [ Registration (SIFT/ORB/Fiducials) ] ──► Affine Matrix (2x3)
[ Electron Microscopy ]──┘                                                   │
                                                                             ├──► [ Coordinate Mapping (x,y) -> (x',y') ]
                                                                             └──► [ Dual-Channel Fusion View (Alpha Overlay) ]

```

---

## Mathematical

ModelCross-modal image alignment is modeled using a 2D Affine Transformation Matrix $M \in \mathbb{R}^{2 \times 3}$:$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$Forward Mapping: Transforms fluorescence pixel coordinates $(x, y)$ to EM target location $(x', y')$.Inverse Mapping: Uses $M^{-1}$ (augmented $3 \times 3$) to map EM coordinates back to the fluorescence domain.

---

## Project Architecture
```
micropy-align/
├── src/
│   ├── registration/      # SIFT/ORB automatic & manual fiducial algorithms
│   ├── transform/         # Affine transformations & image warping
│   ├── mapping/           # Forward and inverse coordinate conversion
│   └── visualization/     # Marker overlays & opacity blending
├── tests/                 # Automated pytest test suite
├── examples/              # Standalone CLI python scripts
├── Dockerfile             # Containerized deployment spec
└── app.py                 # Streamlit web UI
```

---

## Installation & Quick Start
Local Setup
git clone [https://github.com/Yooyoo24/micropy-align.git](https://github.com/Yooyoo24/micropy-align.git)
cd micropy-align
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run Streamlit Web Application
streamlit run app.py

Run via Docker
docker build -t micropy-align .
docker run -p 8501:8501 micropy-align

---

## Testing
Run unit tests to verify coordinate transformations:
pytest

---

## Limitations
Extreme Modality Contrast: If cross-modal contrast varies too drastically for SIFT/ORB feature matching, manual fiducial points are recommended.

2D Rigid/Affine Assumption: The engine assumes planar 2D transformations; non-linear local deformations (e.g., sample shrinkage) require non-rigid registration (e.g., B-spline).

---

## License
Distributed under the MIT License.
