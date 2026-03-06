# Human Activity Recognition using Hidden Markov Models

## Group Members
- **Mitali** — Samsung SM-X216B Tablet, Android, 50Hz
- **Aubert** — iPhone 13, iOS, 50Hz

## Project Structure
- `dataset/` — 53 labeled CSV files (Accelerometer + Gyroscope)
- `HMM_Activity_Recognition.ipynb` — Complete Jupyter notebook
- `hmm_activity_recognition.py` — Python script version
- `HMM_Report.pdf` — Project report (4–5 pages)

## Activities Recorded
| Activity | Samples | Total Duration |
|----------|---------|---------------|
| Standing | 14 | 104.4s |
| Walking | 13 | 96.0s |
| Jumping | 13 | 93.8s |
| Still | 13 | 95.6s |

## Model
- **Type**: Gaussian Hidden Markov Model (implemented from scratch with NumPy)
- **Features**: 37 time-domain and frequency-domain features per window
- **Training**: Baum-Welch algorithm with log-likelihood convergence check
- **Decoding**: Viterbi algorithm (log-space)

## Results
- Training accuracy: **98.09%**
- Test accuracy: **100%** (on 2 unseen samples)

## How to Run
```bash
pip install numpy scipy pandas matplotlib seaborn
jupyter notebook HMM_Activity_Recognition.ipynb
```
