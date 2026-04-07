# Locomotor Resilience Analysis — Postural Control During Gait Perturbations

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.0%2B-276DC3?logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**Master IEAP — Stage M1/M2 Research Project**  
Analysis pipeline for quantifying locomotor resilience using Time-Delay Embedding (TDE) and state-space reconstruction from 3D motion capture data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies](#technologies)
- [License](#license)
- [Contact & Citation](#contact--citation)

---

## 🎯 Overview

This project implements a complete **locomotor resilience analysis pipeline** to quantify postural control responses to mechanical gait perturbations. The analysis combines:

- **Empirical Mode Decomposition (EMD)** for signal preprocessing
- **Time-Delay Embedding (TDE)** for state-space reconstruction
- **Distance-to-attractor metrics** for stability quantification
- **Recovery time analysis** across stability zones (T1σ, T2σ, T3σ)

**Study Design:**
- **N = 7 participants** (5 Young Adults, 2 Aging Adults)
- **2 perturbation conditions** (Slow, Fast) during treadmill walking
- **CODAMOTION 3D motion capture** (100 Hz, sacrum markers)
- **Validation protocol** for M2 internship (BeatMove-CODA synchronization)

**Key Findings:**
- Fast perturbations induced significantly longer recovery times (1.22s ± 0.34s) compared to slow perturbations (0.00s ± 0.00s, instant recovery)
- State-space trajectories revealed distinct attractor dynamics between perturbation conditions
- Synchronization protocol validated with R² > 0.99, RMSE < 160ms

---

## ✨ Features

- ✅ **Automated EMD preprocessing** — Adaptive noise reduction for biomechanical signals
- ✅ **TDE parameter optimization** — Mutual Information (τ) + False Nearest Neighbors (m)
- ✅ **3D state-space reconstruction** — Phase-space attractor visualization
- ✅ **Multi-threshold stability analysis** — T1σ, T2σ, T3σ zones
- ✅ **Interactive 3D visualizations** — Plotly-based trajectory rendering
- ✅ **Cross-platform compatibility** — Windows, macOS, Linux
- ✅ **Reproducible research** — Jupyter notebooks + R Markdown reports

---

## 🔧 Installation

### Prerequisites

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **R 4.0+** ([Download](https://www.r-project.org/))
- **Git** ([Download](https://git-scm.com/downloads))

### Clone Repository

```bash
git clone https://github.com/VictorSal0student/resilience-postural-salvat-victor
cd resilience-postural-salvat-victor
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `numpy`, `scipy`, `pandas` — Core numerical computing
- `EMD-signal` — Empirical Mode Decomposition
- `matplotlib`, `seaborn` — Visualization
- `h5py` — MATLAB data loading
- `jupyter` — Interactive notebooks

### R Dependencies

```r
install.packages(c("plotly", "readr", "dplyr", "rmarkdown", "knitr"))
```

---

## 🚀 Usage

### Quick Start — Single Participant Analysis

```bash
# Run complete analysis pipeline
jupyter nbconvert --execute notebooks/01_single_participant.ipynb

# Generate HTML report
Rscript -e "rmarkdown::render('salvat.victor.Rmd')"
```

**Output:**
- `results/figures/` — 11 publication-ready figures
- `results/metrics/` — CSV files with recovery metrics
- `salvat.victor.html` — Complete interactive report

### Pipeline Notebooks

1. **`01_single_participant.ipynb`** — Full pipeline (Participant 004)
   - EMD preprocessing
   - TDE parameter selection
   - State-space reconstruction
   - Recovery metrics computation
   - Figure generation

2. **`02_group_analysis.ipynb`** — Statistical comparison (N=7)
   - Young vs Aging groups
   - Slow vs Fast perturbations
   - ANOVA + post-hoc tests

3. **`03_beatmove_sync.ipynb`** — Synchronization validation
   - IMU-MOCAP alignment protocol
   - R² > 0.99, RMSE < 160ms
   - M2 internship validation

### R Markdown Report

```r
# Open RStudio
# File → Open → salvat.victor.Rmd
# Click "Knit to HTML"
```

**Interactive 3D figures** generated via `sources/r/torus_3d.R`

---

## 📁 Project Structure

```
resilience-postural-salvat-victor/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
├── salvat.victor.Rmd             # R Markdown report
├── salvat.victor.html            # Generated HTML report
│
├── data/                         # Motion capture datasets
│   ├── README.md                 # Dataset documentation
│   ├── antoine/                  # Main study (N=7, CODA)
│   │   └── .gitkeep              # (Large .mat files excluded)
│   └── premanip/                 # Validation protocol
│       ├── beatmove/             # IMU sensor data (CSV)
│       └── coda/                 # Motion capture sync files
│
├── notebooks/                    # Jupyter analysis pipelines
│   ├── 01_single_participant.ipynb
│   ├── 02_group_analysis.ipynb
│   └── 03_beatmove_sync.ipynb
│
├── results/
│   ├── figures/                  # Publication-ready plots
│   │   ├── pipeline_overview.png
│   │   ├── 004_recovery_curves.png
│   │   ├── phase_space_*.csv     # 3D trajectory data
│   │   └── ...
│   └── metrics/                  # Quantitative outputs
│       ├── recovery_metrics_004.csv
│       └── sync_validation.csv
│
└── sources/                      # Reusable code modules
    ├── python/
    │   ├── preprocessing.py      # EMD, filtering
    │   ├── tde_analysis.py       # Embedding, MI, FNN
    │   ├── state_space.py        # Distance metrics
    │   ├── perturbation_detection.py
    │   └── ...
    ├── r/
    │   ├── torus_3d.R            # Interactive 3D plots
    │   ├── group_comparison.R
    │   └── ...
    └── matlab/
        ├── load_coda_data.m
        └── extract_markers.m
```

---

## 📊 Dataset

### ⚠️ GitHub Size Limit (<100 MB)

**Large .mat files (~80 MB each) are NOT included in this repository.**

### Complete Dataset (Available Upon Request)

- **Participants:** N = 15 (7 analyzed, 8 pending)
- **Total size:** ~1.2 GB
- **Format:** MATLAB v7.3 HDF5 (CODAMOTION)
- **Contact:** victor.salvat@etu.umontpellier.fr
- **Storage:** Google Drive
[link: https://drive.google.com/drive/folders/1kY2Lvfu3Yd8_76CB6gbowGigxCX-dyJk?usp=drive_link]

### Required Files for Reproduction

To execute the complete analysis pipeline:

```
data/antoine/004_Cued.mat          (~80 MB) — Main participant
data/premanip/coda/AD01_016*.mat   (~80 MB) — Sync validation Session 1
data/premanip/coda/AD01_017*.mat   (~80 MB) — Sync validation Session 2
```

### Included in Repository

- **`data/premanip/beatmove/`** — IMU sensor CSV files (lightweight)
- Sessions 1 & 2: accelerometer, gyroscope, timestamp data
- Usable for synchronization protocol validation

### Reproduction Instructions

1. **Request dataset files** (contact above)
2. **Place .mat files** in `data/antoine/` and `data/premanip/coda/`
3. **Run notebooks:**
   ```bash
   jupyter nbconvert --execute notebooks/01_single_participant.ipynb
   ```
4. **Verify outputs** in `results/figures/`

---

## 📈 Results

### Key Metrics (Participant 004)

| Metric | Slow Perturbation | Fast Perturbation |
|--------|-------------------|-------------------|
| **Recovery Time** | 0.00s (instant) | 1.22s |
| **Max Distance** | 22.03 mm | 39.94 mm |
| **Stability Zone** | T1σ maintained | T3σ exceeded |

### Generated Figures

1. **Pipeline Overview** — Flowchart (EMD → TDE → Metrics)
2. **TDE Parameters** — Mutual Information + FNN plots
3. **Recovery Curves** — Distance vs Time with thresholds
4. **3D State-Space (SLOW)** — Interactive Plotly visualization
5. **3D State-Space (FAST)** — Trajectory comparison

**View complete results:** [salvat.victor.html](https://github.com/VictorSal0student/resilience-postural-salvat-victor/blob/main/salvat.victor.html)

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.10+, R 4.0+, MATLAB R2021a |
| **Data Processing** | NumPy, SciPy, Pandas, EMD-signal |
| **Visualization** | Matplotlib, Seaborn, Plotly (R) |
| **Notebooks** | Jupyter, R Markdown, Knitr |
| **Hardware** | CODAMOTION Cx1 (3D motion capture, 100 Hz) |
| **IMU Sensors** | BeatMove (accelerometer, gyroscope) |
| **Version Control** | Git, GitHub |

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Victor SALVAT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Contact & Citation

**Author:** Victor SALVAT  
**Affiliation:** Master IEAP, Université de Montpellier  
**Email:** victor.salvat@etu.umontpellier.fr  
**GitHub:** [@VictorSal0student](https://github.com/VictorSal0student)  
**Defense Date:** April 21, 2026

### Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{salvat2026resilience,
  author  = {Victor SALVAT},
  title   = {Locomotor Resilience Analysis Using Time-Delay Embedding and State-Space Reconstruction},
  school  = {Université de Montpellier},
  year    = {2026},
  type    = {Master's Thesis},
  url     = {https://github.com/VictorSal0student/resilience-postural-salvat-victor}
}
```

---

**📄 Full Report:** [salvat.victor.html](salvat.victor.html)  
**🔬 Interactive Visualizations:** Run `salvat.victor.Rmd` in RStudio  
**📊 Dataset Access:** Contact victor.salvat@etu.umontpellier.fr

---

*Last updated: April 7, 2026*
