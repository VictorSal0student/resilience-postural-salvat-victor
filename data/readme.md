# Data Directory

## ⚠️ GitHub Size Limit (<100 MB)

**Large .mat files (~80 MB each) are NOT included in this repository to comply with GitHub's file size recommendations.**

---

## 📊 Complete Dataset (Available Upon Request)

**Study Participants:**
- **Total:** N = 15
- **Analyzed:** N = 7 (5 Young Adults, 2 Aging Adults)

**Dataset Characteristics:**
- **Total size:** > 1.2 GB (uncompressed)
- **File format:** MATLAB v7.3 HDF5 (CODAMOTION Cx1 export)
- **Sampling rate:** 100 Hz
- **Markers:** 4 sacrum markers (3D coordinates)
- **Conditions:** Baseline + Slow/Fast perturbations

**Access:**
- **Contact:** victor.salvat@etu.umontpellier.fr
- **Storage:** Google Drive [link: https://drive.google.com/drive/folders/1kY2Lvfu3Yd8_76CB6gbowGigxCX-dyJk?usp=sharing]
- **License:** Data available for academic research upon reasonable request

---

## 📁 Required Files for Complete Reproduction

To execute the **full analysis pipeline** (`notebooks/01_single_participant.ipynb`, `02_group_analysis.ipynb`), you need:

### Main Study (N=7)

```
data/antoine/004_Cued.mat          (~80 MB)
data/antoine/005_Cued.mat          (~80 MB)
data/antoine/006_Cued.mat          (~80 MB)
data/antoine/008_Cued.mat          (~80 MB)
data/antoine/009_Cued.mat          (~80 MB)
data/antoine/010_Cued.mat          (~80 MB)
data/antoine/017_Cued.mat          (~80 MB)
data/antoine/015_Cued.mat          (~80 MB)
```

**Participant Details:**
- **004, 006, 008, 009, 017** — Young Adults (18-30 years)
- **006, 010** — Aging Adults (60+ years)

### Synchronization Validation (M2 Protocol)

```
data/premanip/coda/AD01_016_2026_3_16.mat  (~41 MB)
data/premanip/coda/AD01_017_2026_3_16.mat  (~41 MB)
```

**Purpose:** BeatMove IMU ↔ CODAMOTION synchronization validation  
**Used in:** `notebooks/03_beatmove_sync.ipynb`

---

## ✅ Data Included in Repository

### Lightweight IMU Sensor Data

**Location:** `data/premanip/beatmove/`

**Sessions:**
- **Session 1/** — 16 CSV files (left/right ankle accelerometer + gyroscope)
- **Session 2/** — 17 CSV files (left/right ankle accelerometer + gyroscope)

**Format:** CSV with columns `phoneTimeMs, aX, aY, aZ, gX, gY, gZ`

**Size:** ~500 KB total (acceptable for GitHub)

**Usage:**
- Synchronization protocol validation
- Step detection algorithm testing
- IMU-MOCAP alignment verification

---

## 🔄 Reproduction Instructions

### Step 1: Request Dataset

**Email:** victor.salvat@etu.umontpellier.fr 
**Subject:** "Request Locomotor Resilience Dataset"  
**Include:**
- Your affiliation
- Intended use (research/education)
- Required participants (single/all)

### Step 2: Download Files

You will receive a **Google Drive link** with folder structure:

```
locomotor-resilience-dataset/
├── main_study/
│   ├── 004_Cued.mat
│   ├── 005_Cued.mat
│   └── ...
└── synchronization_validation/
    ├── AD01_016_2026_3_16.mat
    └── AD01_017_2026_3_16.mat
```

### Step 3: Place Files in Repository

```bash
# Navigate to repository
cd resilience-postural-salvat-victor/

# Create directories if needed
mkdir -p data/antoine
mkdir -p data/premanip/coda

# Copy .mat files
cp ~/Downloads/004_Cued.mat data/antoine/
cp ~/Downloads/AD01_016_2026_3_16.mat data/premanip/coda/
# ... (repeat for all files)
```

### Step 4: Verify Installation

```bash
# Check files are in place
ls -lh data/antoine/
ls -lh data/premanip/coda/

# Expected output:
# data/antoine/004_Cued.mat  (~80 MB)
# data/premanip/coda/AD01_016_2026_3_16.mat  (~41 MB)
```

### Step 5: Run Analysis

```bash
# Execute single participant pipeline
jupyter nbconvert --execute notebooks/01_single_participant.ipynb

# Check outputs
ls results/figures/
# Expected: 004_recovery_curves.png, phase_space_SLOW.csv, etc.
```

---

## 📋 File Naming Convention

**Main study files:**
```
<PARTICIPANT_ID>_Cued.mat
```
- `PARTICIPANT_ID`: 3-digit code (001-299 = Young, 301-399 = Aging)
- `Cued`: Experimental protocol (cued perturbations)

**Synchronization files:**
```
AD01_<SESSION>_YYYY_M_DD.mat
```
- `AD01`: Participant code (premanipulation protocol)
- `SESSION`: 016 (Session 1), 017 (Session 2)
- Date: March 16, 2026

---

## 🔒 Data Privacy & Ethics

- **Anonymized:** All participant identifiers removed
- **Ethics approval:** Protocol approved by institutional review board
- **Consent:** Participants signed informed consent for data sharing
- **GDPR compliant:** No personal health information included

---

## 🛠️ Data Loading Examples

### Python (h5py)

```python
import h5py
import numpy as np

# Load MATLAB v7.3 file
with h5py.File('data/antoine/004_Cued.mat', 'r') as f:
    markers = f['Marker']['Dos01'][()]  # Sacrum marker 1
    # Shape: (n_frames, 3) — X, Y, Z coordinates
```

### MATLAB

```matlab
% Load .mat file
load('data/antoine/004_Cued.mat');

% Extract markers
marker1 = Marker.Dos01;  % (n_frames x 3)
marker2 = Marker.Dos02;
```

### R (not recommended, use Python preprocessing)

```r
# Install rhdf5
# BiocManager::install("rhdf5")

library(rhdf5)
data <- h5read("data/antoine/004_Cued.mat", "Marker/Dos01")
```

---

## 📞 Support

**Questions about dataset structure?**  
→ Open an issue on GitHub: [Issues](https://github.com/VictorSal0student/resilience-postural-salvat-victor/issues)

**Need additional participants?**  
→ Contact: victor.salvat@etu.montpellier.fr

**Technical problems with .mat loading?**  
→ Check Python version (3.10+) and `h5py` installation

---

## 📚 References

**Motion Capture System:**
- CODAMOTION Cx1 (Charnwood Dynamics Ltd.)
- Specification: https://codamotion.com/

**IMU Sensors:**
- BeatMove (custom Android app)
- Sampling: 100 Hz accelerometer + gyroscope

---

*Last updated: April 7, 2026*
