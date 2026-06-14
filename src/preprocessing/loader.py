"""
CardioVision — ECG Data Loader
Supports MIT-BIH (wfdb), CSV, and EDF formats.
Segments raw signals into individual heartbeats for classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RAW_DATA_DIR, SAMPLING_RATE, BEAT_WINDOW, BEAT_BEFORE, BEAT_AFTER,
    ANNOTATION_MAP, MITBIH_RECORDS,
)
from src.preprocessing.filters import apply_all_filters
from src.preprocessing.imputer import handle_missing_values
from src.preprocessing.auditor import SignalAuditor, AuditReport


class ECGLoader:
    """
    Unified loader for ECG data from MIT-BIH, CSV, and EDF sources.
    Handles loading, auditing, preprocessing, and beat segmentation.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        sampling_rate: float = SAMPLING_RATE,
        apply_filters: bool = True,
        apply_imputation: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fs = sampling_rate
        self.apply_filters = apply_filters
        self.apply_imputation = apply_imputation
        self.auditor = SignalAuditor(expected_fs=self.fs)

    # ─────────────────── MIT-BIH Loading ───────────────────

    def download_mitbih(self, records: Optional[List[str]] = None) -> None:
        """Download MIT-BIH records using wfdb."""
        import wfdb

        records = records or MITBIH_RECORDS
        dl_dir = str(self.data_dir / "mitbih")
        os.makedirs(dl_dir, exist_ok=True)

        for rec_id in records:
            rec_path = os.path.join(dl_dir, rec_id)
            if os.path.exists(rec_path + ".dat"):
                continue
            try:
                wfdb.dl_database("mitdb", dl_dir=dl_dir, records=[rec_id])
                print(f"  ✓ Downloaded record {rec_id}")
            except Exception as e:
                warnings.warn(f"  ✗ Failed to download record {rec_id}: {e}")

    def load_mitbih_record(self, record_id: str) -> Dict:
        """
        Load a single MIT-BIH record.

        Returns
        -------
        dict with keys: signal, annotations, fs, record_id, audit_report
        """
        import wfdb

        rec_path = str(self.data_dir / "mitbih" / record_id)
        record = wfdb.rdrecord(rec_path)
        annotation = wfdb.rdann(rec_path, "atr")

        # Use channel 0 (MLII lead)
        signal = record.p_signal[:, 0].astype(np.float64)
        fs = record.fs

        # Audit
        audit = self.auditor.audit(signal, record_id=record_id, actual_fs=fs)

        # Impute missing values
        if self.apply_imputation:
            signal, imp_info = handle_missing_values(signal)

        # Filter
        if self.apply_filters:
            signal = apply_all_filters(signal, fs=fs)

        return {
            "signal": signal,
            "ann_samples": annotation.sample,
            "ann_symbols": annotation.symbol,
            "fs": fs,
            "record_id": record_id,
            "audit_report": audit,
        }

    def load_all_mitbih(
        self,
        records: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[AuditReport]]:
        """
        Load and segment all MIT-BIH records into beat arrays.

        Returns
        -------
        X : (N, BEAT_WINDOW) array of beat segments.
        y : (N,) array of class labels.
        record_ids : (N,) array of string record IDs (patient IDs).
        reports : List of AuditReport for each record.
        """
        records = records or MITBIH_RECORDS
        all_beats = []
        all_labels = []
        all_record_ids = []
        all_reports = []

        for rec_id in records:
            try:
                data = self.load_mitbih_record(rec_id)
            except Exception as e:
                warnings.warn(f"Skipping record {rec_id}: {e}")
                continue

            all_reports.append(data["audit_report"])
            signal = data["signal"]
            ann_samples = data["ann_samples"]
            ann_symbols = data["ann_symbols"]

            beats, labels = self._segment_beats(
                signal, ann_samples, ann_symbols
            )
            if len(beats) > 0:
                all_beats.append(beats)
                all_labels.append(labels)
                all_record_ids.extend([rec_id] * len(labels))

        if not all_beats:
            return np.array([]), np.array([]), np.array([]), all_reports

        X = np.vstack(all_beats)
        y = np.concatenate(all_labels)
        record_ids = np.array(all_record_ids)
        return X, y, record_ids, all_reports

    # ─────────────────── CSV Loading ───────────────────

    def load_csv(
        self,
        filepath: str,
        signal_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Dict:
        """
        Load ECG signal from a CSV file.

        Supports two formats:
        1. Single-column: each row is one sample.
        2. Multi-column: specify signal_col for the ECG channel.
        """
        df = pd.read_csv(filepath)

        if signal_col and signal_col in df.columns:
            signal = df[signal_col].values.astype(np.float64)
        elif len(df.columns) == 1:
            signal = df.iloc[:, 0].values.astype(np.float64)
        else:
            # Try to auto-detect: use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in CSV.")
            signal = df[numeric_cols[0]].values.astype(np.float64)

        audit = self.auditor.audit(signal, record_id=Path(filepath).stem)

        if self.apply_imputation:
            signal, _ = handle_missing_values(signal)
        if self.apply_filters:
            signal = apply_all_filters(signal, fs=self.fs)

        return {
            "signal": signal,
            "fs": self.fs,
            "record_id": Path(filepath).stem,
            "audit_report": audit,
        }

    # ─────────────────── EDF Loading ───────────────────

    def load_edf(
        self,
        filepath: str,
        channel: int = 0,
    ) -> Dict:
        """
        Load ECG signal from an EDF file.

        Parameters
        ----------
        filepath : Path to .edf file.
        channel : Channel index to read (default: 0).
        """
        import pyedflib

        reader = pyedflib.EdfReader(filepath)
        try:
            n_channels = reader.signals_in_file
            if channel >= n_channels:
                raise ValueError(
                    f"Channel {channel} requested but file has {n_channels} channels."
                )

            signal = reader.readSignal(channel).astype(np.float64)
            fs = reader.getSampleFrequency(channel)
        finally:
            reader.close()

        audit = self.auditor.audit(signal, record_id=Path(filepath).stem, actual_fs=fs)

        if self.apply_imputation:
            signal, _ = handle_missing_values(signal)
        if self.apply_filters:
            signal = apply_all_filters(signal, fs=fs)

        return {
            "signal": signal,
            "fs": fs,
            "record_id": Path(filepath).stem,
            "audit_report": audit,
        }

    # ─────────────────── Beat Segmentation ───────────────────

    def _segment_beats(
        self,
        signal: np.ndarray,
        ann_samples: np.ndarray,
        ann_symbols: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment a continuous ECG signal into individual beats
        centered on R-peak annotations.

        Returns
        -------
        beats : (N, BEAT_WINDOW) array of beat segments.
        labels : (N,) array of integer class labels.
        """
        beats = []
        labels = []

        for idx, symbol in zip(ann_samples, ann_symbols):
            if symbol not in ANNOTATION_MAP:
                continue

            start = idx - BEAT_BEFORE
            end = idx + BEAT_AFTER

            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]
            if len(beat) != BEAT_WINDOW:
                continue

            # Z-score normalize each beat
            std = np.std(beat)
            if std > 1e-8:
                beat = (beat - np.mean(beat)) / std
            else:
                continue  # Skip flat beats

            beats.append(beat)
            labels.append(ANNOTATION_MAP[symbol])

        if not beats:
            return np.array([]), np.array([])

        return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

    def segment_uploaded_signal(
        self,
        signal: np.ndarray,
        fs: float = SAMPLING_RATE,
    ) -> np.ndarray:
        """
        Segment an uploaded signal (without annotations) into fixed-length
        windows using simple R-peak detection.

        Returns
        -------
        segments : (N, BEAT_WINDOW) array of segments.
        """
        from scipy.signal import find_peaks

        # Simple R-peak detection: find peaks above 0.6 * max amplitude
        height_threshold = 0.6 * np.max(np.abs(signal))
        min_distance = int(0.4 * fs)  # Minimum 0.4s between peaks

        peaks, _ = find_peaks(
            signal, height=height_threshold, distance=min_distance
        )

        if len(peaks) == 0:
            # Fall back to fixed-window segmentation
            n_segments = len(signal) // BEAT_WINDOW
            if n_segments == 0:
                return np.array([])
            segments = signal[:n_segments * BEAT_WINDOW].reshape(n_segments, BEAT_WINDOW)
            # Normalize each segment
            stds = np.std(segments, axis=1, keepdims=True)
            stds[stds < 1e-8] = 1.0
            segments = (segments - np.mean(segments, axis=1, keepdims=True)) / stds
            return segments.astype(np.float32)

        segments = []
        for peak in peaks:
            start = peak - BEAT_BEFORE
            end = peak + BEAT_AFTER
            if start < 0 or end > len(signal):
                continue
            beat = signal[start:end]
            if len(beat) != BEAT_WINDOW:
                continue
            std = np.std(beat)
            if std > 1e-8:
                beat = (beat - np.mean(beat)) / std
                segments.append(beat)

        if not segments:
            return np.array([])

        return np.array(segments, dtype=np.float32)
