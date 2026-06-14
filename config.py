"""
CardioVision — Global Configuration
All hyperparameters, paths, and system settings in one place.
"""

import os
import torch
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Device ───────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# ──────────────────────────── Signal ───────────────────────────
SAMPLING_RATE = 360          # MIT-BIH default (Hz)
BEAT_WINDOW = 360            # samples per beat segment (1 second)
BEAT_BEFORE = 180            # samples before R-peak
BEAT_AFTER = 180             # samples after R-peak

# ──────────────────────────── Filters ──────────────────────────
BUTTER_LOWCUT = 0.5          # Hz — remove baseline wander
BUTTER_HIGHCUT = 40.0        # Hz — remove high-frequency noise
BUTTER_ORDER = 4
NOTCH_FREQ = 60.0            # Hz — power-line interference
NOTCH_QUALITY = 30.0         # Quality factor

# ──────────────────────────── Classes ──────────────────────────
CLASS_NAMES = ["Normal", "AFib", "PVC", "Global"]
NUM_CLASSES = len(CLASS_NAMES)

# MIT-BIH annotation symbol → class index
ANNOTATION_MAP = {
    # Normal beats
    "N": 0, "L": 0, "R": 0,
    # Supraventricular / AFib-related
    "A": 1, "a": 1, "J": 1, "S": 1, "e": 1, "j": 1,
    # Premature Ventricular Contraction
    "V": 2, "E": 2,
    # Global / Other
    "/": 3, "f": 3, "F": 3, "Q": 3, "!": 3,
}

# ──────────────────────────── Model ────────────────────────────
# CNN
CNN_CHANNELS = [1, 64, 128, 256]
CNN_KERNEL_SIZES = [5, 5, 3]
CNN_POOL_SIZE = 2
CNN_DROPOUT = 0.3

# LSTM
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.3

# Fully Connected
FC_HIDDEN = 128
FC_DROPOUT = 0.5

# ──────────────────────────── Training ─────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ──────────────────────────── MIT-BIH ─────────────────────────
# Records to use (full 48-record MIT-BIH Arrhythmia Database)
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219", "220",
    "221", "222", "223", "228", "230", "231", "232", "233", "234",
]
