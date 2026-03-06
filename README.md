# Modeling Human Activity States Using Hidden Markov Models

## Group Members & Devices

| Member | Device | OS | Sampling Rate |
|--------|--------|----|--------------|
| Mitali | Samsung SM-X216B Tablet | Android | 50 Hz (20 ms) |
| Aubert | iPhone 13 | iOS | 50 Hz (20 ms) |

> **Sampling rate harmonisation:** Both devices were configured in the Sensor Logger app to record at 50 Hz. Before merging accelerometer and gyroscope streams per recording, timestamps are aligned using `pandas.merge_asof` with a 10 ms tolerance, ensuring cross-device consistency.

---

## Repository Structure

```
HMM-Human-Activity-Recognition/
├── dataset/
│   └── dataset/
│       ├── Jumping/      # 13 × Accelerometer + 13 × Gyroscope CSV files
│       ├── Standing/     # 14 × Accelerometer + 14 × Gyroscope CSV files
│       ├── Still/        # 12 × Accelerometer + 12 × Gyroscope CSV files
│       └── Walking/      # 13 × Accelerometer + 13 × Gyroscope CSV files
├── figures/              # All generated visualisation plots
├── HMM_Activity_Recognition.ipynb   # Main Jupyter notebook
├── hmm_activity_recognition.py      # Equivalent Python script
├── HMM_Report-1.pdf                 # Project report (4–5 pages)
├── requirements.txt
└── .gitignore
```

---

## Data Collection

52 labelled recording pairs (Accelerometer + Gyroscope) were collected across four activities using the **Sensor Logger** app.

| Activity | Total Samples | Mitali | Aubert | Min Duration |
|----------|:---:|:---:|:---:|:---:|
| Standing | 14 | 7 | 7 | 5 s |
| Walking  | 13 | 7 | 6 | 5 s |
| Jumping  | 13 | 7 | 6 | 5 s |
| Still    | 12 | 6 | 6 | 5 s |
| **Total**| **52** | **27** | **25** | — |

Each file contains: `time`, `seconds_elapsed`, `x`, `y`, `z` columns at 50 Hz.

---

## Windowing Strategy

| Parameter | Value | Justification |
|-----------|-------|--------------|
| Sampling rate | 50 Hz | Set explicitly on both devices |
| Window size | 50 samples (1 s) | At 50 Hz, 50 samples = 1 second — long enough for a complete gait cycle (~0.5 s/step for walking) and to resolve frequencies down to 1 Hz via FFT |
| Overlap | 25 samples (50%) | Reduces information loss at window boundaries; standard practice for activity recognition |

---

## Feature Extraction

37 features per window, drawn from both domains:

**Time-domain (per axis + magnitude):**
- Mean, Standard Deviation, RMS — capture signal energy and variability
- Signal Magnitude Area (SMA) — overall motion intensity
- Axis correlations (XY, XZ) — coordinated vs independent movement

**Frequency-domain (FFT-derived, per accelerometer axis + magnitude):**
- Dominant frequency — walking cadence ≈ 2 Hz, jumping ≈ 1–2 Hz
- FFT peak magnitude — strength of dominant periodic component
- Spectral energy — total power distinguishes high/low energy activities

All features are normalised using **Z-score normalisation** (zero mean, unit variance) to prevent high-magnitude features from dominating the model.

---

## HMM Implementation

- **Type**: Gaussian HMM implemented from scratch using NumPy
- **Hidden states (Z)**: Standing, Walking, Jumping, Still
- **Observations (X)**: 37-dimensional normalised feature vectors
- **Transition matrix (A)**: 4 × 4, learned via Baum–Welch
- **Emission model (B)**: Multivariate Gaussian per state (mean + covariance)
- **Initial probabilities (π)**: Estimated from training label frequencies
- **Training**: Baum–Welch EM algorithm with log-likelihood convergence check (`|ΔLL| < 1e-4`)
- **Decoding**: Viterbi algorithm (log-space, with backpointer)

---

## Evaluation Results (Unseen Test Data)

Two test recordings were withheld (one per participant, never seen during training):
- Mitali — Walking sample 7
- Aubert — Jumping sample 6

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|-----------------|:-----------------:|:-----------:|:-----------:|:----------------:|
| Standing | — | — | — | — |
| Walking  | — | — | — | — |
| Jumping  | — | — | — | — |
| Still    | — | — | — | — |

> Run the notebook to populate this table with live results.

- **Training set accuracy**: 98.09%
- **Test set accuracy**: 100% (2 unseen samples)

---

## Task Allocation

| Task | Mitali | Aubert |
|------|:------:|:------:|
| Data collection (own recordings) | ✓ | ✓ |
| Dataset organisation & upload | ✓ | ✓ |
| Data loading & preprocessing | ✓ | |
| Feature extraction (time-domain) | ✓ | |
| Feature extraction (frequency-domain) | ✓ | |
| HMM class implementation (Baum–Welch, Viterbi) | ✓ | |
| Model training & convergence analysis | | ✓ |
| Evaluation metrics & confusion matrix | | ✓ |
| Visualisations (transition/emission/decoded) | | ✓ |
| Report writing | ✓ | ✓ |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook HMM_Activity_Recognition.ipynb
```

All output figures are saved automatically to the `figures/` directory.
