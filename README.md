# Activity Recognition from Wrist-Worn Sensors

Wrist-worn sensor data pipeline and ML classifier for recognising 5 daily activities — smoking, typing, idle, cooking, and exercising — using Phyphox accelerometer and gyroscope data.

---

## Overview

This project investigates whether a machine learning model can infer a person's current activity purely from the motion of their wrist. Using the [Phyphox](https://phyphox.org/) app on a wrist-mounted smartphone, we collected accelerometer, gyroscope, and linear accelerometer data across five activities that span a wide behavioral spectrum — from complete stillness to vigorous physical movement.

The dataset is collected by 3 participants, each recording 8 sessions per activity, yielding 120 labeled sessions total. Features are extracted from sliding windows over the raw time-series data and fed into a classical ML classifier.

---

## Activities

| Activity | Description |
|---|---|
| **Smoking** | Repeated hand-to-mouth gesture; slow, periodic arc motion |
| **Typing + Mouse** | Desk work; rapid low-amplitude wrist vibrations from keystrokes |
| **Idle** | Seated at rest, hand not engaged; near-zero signal baseline |
| **Cooking** | Stirring, chopping, reaching; freeform high-variance arm motion |
| **Exercising** | Brisk walking or jogging; strong periodic full-body movement |

---

## Sensors

All data collected via Phyphox with the phone strapped to the **dominant wrist**.

| Sensor | Columns | Why |
|---|---|---|
| Accelerometer | `acc_x`, `acc_y`, `acc_z` | Linear movement and gravity |
| Gyroscope | `gyr_x`, `gyr_y`, `gyr_z` | Rotational velocity of the wrist |
| Linear Accelerometer | `lin_x`, `lin_y`, `lin_z` | Motion with gravity removed |

---

## Repository Structure

```
ml-activity-recognition/
│
├── data/
│   └── raw/
│       ├── alice/
│       │   ├── smoking/
│       │   │   ├── session_01/
│       │   │   │   ├── Accelerometer.csv
│       │   │   │   ├── Gyroscope.csv
│       │   │   │   ├── Linear Accelerometer.csv
│       │   │   │   └── meta
│       │   │   └── ...session_08/
│       │   ├── typing/
│       │   ├── idle/
│       │   ├── cooking/
│       │   └── exercising/
│       ├── bob/
│       └── charlie/
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Explore raw sensor data
│   ├── 02_preprocessing.ipynb  ← Windowing and feature extraction
│   └── 03_modelling.ipynb      ← Train, evaluate, confusion matrix
│
├── src/
│   ├── load_data.py            ← Load and merge all sessions into one df
│   ├── preprocess.py           ← Trimming, windowing
│   ├── features.py             ← Feature extraction per window
│   └── train.py                ← Model training and evaluation
│
├── outputs/                    ← Generated files, not committed to git
│   ├── features/
│   │   └── feature_matrix.csv
│   └── models/
│       └── random_forest.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

- **Participants:** 3
- **Activities:** 5
- **Sessions per participant per activity:** 8
- **Total sessions:** 120
- **Recording duration:** ~30 seconds per session
- **Sampling rate:** ~500 Hz
- **Windowing:** 2-second windows, 1-second step (50% overlap)
- **Windows per session:** ~29
- **Total windows (approx):** ~3,480

### Train / Test Split

Data is split **by person**, not randomly, to test generalisation to an unseen individual:

| Split | Person | Sessions | Windows (approx) |
|---|---|---|---|
| Train | alice + bob | 80 | ~2,320 |
| Test | charlie | 40 | ~1,160 |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourteam/ml-activity-recognition.git
cd ml-activity-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data

Place your Phyphox session folders under `data/raw/` following the structure above. Each session folder should contain `Accelerometer.csv`, `Gyroscope.csv`, and `Linear Accelerometer.csv`.

### 4. Run the pipeline

```bash
# Load and inspect raw data
jupyter notebook notebooks/01_eda.ipynb

# Preprocess and extract features
jupyter notebook notebooks/02_preprocessing.ipynb

# Train and evaluate the classifier
jupyter notebook notebooks/03_modelling.ipynb
```

---

## Preprocessing Pipeline

```
Raw CSVs
  → Load all sessions + attach person / activity / session labels
  → Merge 3 sensors on nearest timestamp (merge_asof)
  → Trim first and last 2 seconds per clip
  → Sliding windows (2s window, 1s step)
  → Extract features per window (time + frequency domain)
  → Normalise (StandardScaler fit on train only)
  → Train / test split by person
```

### Features Extracted Per Window

For each axis of each sensor: mean, std, min, max, RMS, range, zero-crossing rate, FFT dominant frequency, FFT max magnitude.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyterlab
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Team

| Name | Role |
|---|---|
| Alice | Data collection, modelling |
| Bob | Data collection, preprocessing |
| Charlie | Data collection, EDA |

---

## Notes

- The timestamp in each Phyphox recording is a **stopwatch from 0**, not a wall-clock time. Activity labels come entirely from the folder structure, not from the data itself.
- Never edit raw CSV files. All transformations happen in code.
- `outputs/` is in `.gitignore` — regenerate locally by running the notebooks in order.
