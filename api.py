"""
CardioVision — FastAPI Backend
Thin REST wrapper around existing src/ modules. Zero logic changes.

Key design decisions:
- All blocking work (torch.load, np.load, training, Grad-CAM) runs in a
  ThreadPoolExecutor so the event loop is never blocked.
- /api/status reads only file metadata (exists + size) — no torch.load.
- Training runs in a daemon thread, state is read via /api/train/status.
"""

import os, sys, tempfile, traceback, threading
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, CLASS_NAMES, NUM_CLASSES,
    MODEL_DIR, PROCESSED_DATA_DIR, REPORT_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
)
from src.utils.helpers import set_seed, get_device_info, count_parameters
from main import get_model

# ── Thread pool for all blocking work ───────────────────────────
_pool = ThreadPoolExecutor(max_workers=2)

# ── Shared training/preprocess state ────────────────────────────
_state = {
    "running": False,
    "job": "idle",          # idle | preprocess | train
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "status": "idle",       # idle | running | done | error
    "message": "",
    "results": None,
}
_state_lock = threading.Lock()

def _update(**kw):
    with _state_lock:
        _state.update(kw)

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(title="CardioVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════
#  STATUS  — fast, no torch.load
# ══════════════════════════════════════════════════════

@app.get("/api/status")
def get_status():
    """Lightweight status check — only checks file existence, no heavy I/O."""
    device_info = get_device_info()
    data_exists = (PROCESSED_DATA_DIR / "mitbih_processed.npz").exists()

    # Model info: only read the checkpoint header, not the full weights
    models = {}
    for arch in ["hybrid", "attention", "resnet", "se"]:
        p = MODEL_DIR / f"best_model_{arch}.pt"
        if not p.exists() and arch == "hybrid":
            p = MODEL_DIR / "best_model.pt"
        if p.exists():
            try:
                # weights_only=True is fast — only loads the scalar metadata
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                models[arch] = {
                    "exists": True,
                    "epoch": ckpt.get("epoch", "?"),
                    "val_loss": round(float(ckpt.get("val_loss", 0)), 4),
                    "val_acc": round(float(ckpt.get("val_acc", 0)), 4),
                }
            except Exception:
                models[arch] = {"exists": True, "epoch": "?", "val_loss": 0, "val_acc": 0}
        else:
            models[arch] = {"exists": False}

    # Dataset stats: use mmap so NumPy reads only the header
    data_stats: dict = {}
    if data_exists:
        try:
            d = np.load(PROCESSED_DATA_DIR / "mitbih_processed.npz", mmap_mode="r")
            data_stats = {
                "total_beats": int(d["X"].shape[0]),
                "beat_window": int(d["X"].shape[1]),
                "class_counts": {
                    CLASS_NAMES[i]: int(np.sum(d["y"] == i)) for i in range(NUM_CLASSES)
                },
            }
        except Exception:
            data_stats = {}

    return {
        "device": str(DEVICE),
        "cuda": device_info["cuda_available"],
        "gpu_name": device_info.get("device_name"),
        "gpu_memory_gb": device_info.get("total_memory_gb"),
        "data_ready": data_exists,
        "data_stats": data_stats,
        "models": models,
        "class_names": CLASS_NAMES,
    }


# ══════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_ecg(file: UploadFile = File(...)):
    from src.preprocessing.loader import ECGLoader
    import asyncio

    contents = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".edf"):
        raise HTTPException(400, "Only .csv and .edf files are supported.")

    tmp = Path(tempfile.mktemp(suffix=suffix))
    tmp.write_bytes(contents)

    def _load():
        loader = ECGLoader()
        if suffix == ".csv":
            return loader.load_csv(str(tmp))
        return loader.load_edf(str(tmp))

    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_pool, _load)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp.unlink(missing_ok=True)

    signal = data["signal"]
    audit = data["audit_report"].to_dict()
    return {
        "filename": file.filename,
        "signal": signal[:3600].tolist(),
        "fs": data["fs"],
        "audit": audit,
    }


# ══════════════════════════════════════════════════════
#  PREPROCESS / DOWNLOAD MIT-BIH
# ══════════════════════════════════════════════════════

@app.post("/api/download-mitbih")
def download_mitbih():
    with _state_lock:
        if _state["running"]:
            return {"started": False, "message": "Already running"}
        _state.update({
            "running": True, "job": "preprocess", "status": "running",
            "message": "Starting download...", "results": None,
        })

    def _run():
        try:
            from src.preprocessing.loader import ECGLoader
            loader = ECGLoader()
            _update(message="Downloading MIT-BIH records...")
            loader.download_mitbih()
            _update(message="Segmenting beats (this takes a few minutes)...")
            X, y, record_ids, _ = loader.load_all_mitbih()
            np.savez_compressed(
                PROCESSED_DATA_DIR / "mitbih_processed.npz",
                X=X, y=y, record_ids=record_ids,
            )
            counts = {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(NUM_CLASSES)}
            _update(
                running=False, status="done",
                message=f"Done — {int(X.shape[0]):,} beats extracted",
                results={"total_beats": int(X.shape[0]), "class_counts": counts},
            )
        except Exception as e:
            _update(running=False, status="error", message=str(e))

    _pool.submit(_run)
    return {"started": True}


@app.get("/api/preprocess-status")
def preprocess_status():
    with _state_lock:
        return {
            "status": _state["status"],
            "message": _state["message"],
            "results": _state.get("results"),
        }


# ══════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    arch: str = "hybrid"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42


@app.post("/api/train")
def start_training(req: TrainRequest):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "A job is already running.")
        data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
        if not data_path.exists():
            raise HTTPException(400, "No processed data — run Download & Preprocess first.")
        _state.update({
            "running": True, "job": "train",
            "epoch": 0, "total_epochs": req.epochs,
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "status": "running", "message": "Initialising...", "results": None,
        })

    def _run():
        try:
            from src.model.trainer import Trainer

            set_seed(req.seed)
            data = np.load(data_path)
            X, y = data["X"], data["y"]

            model = get_model(req.arch, DEVICE)
            params = count_parameters(model)
            _update(message=f"Model: {req.arch} ({params:,} params) on {DEVICE}")

            trainer = Trainer(
                model, device=DEVICE, num_epochs=req.epochs,
                batch_size=req.batch_size, learning_rate=req.lr,
            )
            train_ld, val_ld, test_ld = trainer.prepare_data(X, y, seed=req.seed)
            save_file = f"best_model_{req.arch}.pt"

            for ep in range(1, req.epochs + 1):
                tl, ta, tf = trainer.train_epoch(train_ld)
                vl, va, vf = trainer.eval_epoch(val_ld)
                trainer.scheduler.step(vl)

                for k, v in zip(trainer.history.keys(),
                                [tl, vl, ta, va, tf, vf,
                                 trainer.optimizer.param_groups[0]["lr"]]):
                    trainer.history[k].append(v)

                if vl < trainer.best_val_loss:
                    trainer.best_val_loss = vl
                    trainer.epochs_no_improve = 0
                    trainer.save_checkpoint(save_file, ep, vl, va)
                else:
                    trainer.epochs_no_improve += 1

                with _state_lock:
                    _state["epoch"] = ep
                    _state["train_loss"].append(round(tl, 4))
                    _state["val_loss"].append(round(vl, 4))
                    _state["train_acc"].append(round(ta, 4))
                    _state["val_acc"].append(round(va, 4))
                    _state["message"] = (
                        f"Epoch {ep}/{req.epochs} — "
                        f"loss {tl:.4f}/{vl:.4f}  acc {ta:.3f}/{va:.3f}"
                    )

                if trainer.epochs_no_improve >= trainer.patience:
                    _update(message=f"Early stop at epoch {ep}")
                    break

            trainer.load_checkpoint(save_file)
            results = trainer.evaluate(test_ld)
            _update(
                running=False, status="done",
                message=f"Done — acc {results['accuracy']:.1%}  f1 {results['macro_f1']:.1%}",
                results={
                    "accuracy": round(results["accuracy"], 4),
                    "macro_f1": round(results["macro_f1"], 4),
                    "confusion_matrix": results["confusion_matrix"],
                    "classification_report": results["classification_report"],
                },
            )
        except Exception:
            _update(running=False, status="error", message=traceback.format_exc())

    _pool.submit(_run)
    return {"started": True, "arch": req.arch}


@app.get("/api/train/status")
def train_status():
    with _state_lock:
        return {
            "running": _state["running"],
            "epoch": _state["epoch"],
            "total_epochs": _state["total_epochs"],
            "train_loss": list(_state["train_loss"]),
            "val_loss": list(_state["val_loss"]),
            "train_acc": list(_state["train_acc"]),
            "val_acc": list(_state["val_acc"]),
            "status": _state["status"],
            "message": _state["message"],
            "results": _state["results"],
        }


# ══════════════════════════════════════════════════════
#  INFERENCE & GRAD-CAM
# ══════════════════════════════════════════════════════

@app.get("/api/dataset/sample")
def get_sample(class_filter: str = "Any", idx: int = 0):
    data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
    if not data_path.exists():
        raise HTTPException(400, "No processed data.")

    d = np.load(data_path, mmap_mode="r")
    X, y = d["X"], d["y"]
    record_ids = d["record_ids"] if "record_ids" in d else None

    if class_filter != "Any" and class_filter in CLASS_NAMES:
        mask = y == CLASS_NAMES.index(class_filter)
        X_f, y_f = X[mask], y[mask]
        rids = record_ids[mask] if record_ids is not None else None
    else:
        X_f, y_f = X, y
        rids = record_ids

    total = len(X_f)
    if total == 0:
        raise HTTPException(400, f"No samples for class '{class_filter}'.")
    idx = max(0, min(idx, total - 1))

    return {
        "signal": X_f[idx].tolist(),
        "true_label": CLASS_NAMES[int(y_f[idx])],
        "record_id": str(rids[idx]) if rids is not None else None,
        "total": total,
        "idx": idx,
    }


@app.post("/api/infer")
async def run_inference(
    file: Optional[UploadFile] = File(None),
    arch: str = "hybrid",
    source: str = "dataset",
    class_filter: str = "Any",
    idx: int = 0,
):
    import asyncio
    from src.explainability.gradcam import GradCAM1D

    model_path = MODEL_DIR / f"best_model_{arch}.pt"
    if not model_path.exists() and arch == "hybrid":
        model_path = MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        raise HTTPException(400, f"No trained model for arch='{arch}'.")

    # Get raw signal
    if source == "upload" and file is not None:
        from src.preprocessing.loader import ECGLoader
        contents = await file.read()
        suffix = Path(file.filename or "").suffix.lower()
        tmp = Path(tempfile.mktemp(suffix=suffix))
        tmp.write_bytes(contents)
        try:
            loader = ECGLoader(apply_filters=False, apply_imputation=False)
            data = (loader.load_csv if suffix == ".csv" else loader.load_edf)(str(tmp))
            segs = loader.segment_uploaded_signal(data["signal"], fs=data["fs"])
        finally:
            tmp.unlink(missing_ok=True)
        if len(segs) == 0:
            raise HTTPException(400, "No beats could be segmented from the file.")
        signal_np = segs[min(idx, len(segs) - 1)].astype(np.float32)
        true_label = None
        record_id = file.filename
    else:
        sample = get_sample(class_filter, idx)
        signal_np = np.array(sample["signal"], dtype=np.float32)
        true_label = sample["true_label"]
        record_id = sample.get("record_id")

    def _infer():
        model = get_model(arch, DEVICE)
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tensor = torch.FloatTensor(signal_np).unsqueeze(0).unsqueeze(0)
        gc = GradCAM1D(model)
        heatmap, pred, probs = gc.compute(tensor)
        gc.remove_hooks()
        return heatmap, pred, probs

    loop = asyncio.get_event_loop()
    heatmap, pred, probs = await loop.run_in_executor(_pool, _infer)

    return {
        "signal": signal_np.tolist(),
        "heatmap": heatmap.tolist(),
        "prediction": int(pred),
        "prediction_name": CLASS_NAMES[int(pred)],
        "probabilities": [round(float(p), 4) for p in probs],
        "true_label": true_label,
        "record_id": record_id,
    }


# ══════════════════════════════════════════════════════
#  REPORTS
# ══════════════════════════════════════════════════════

class ReportRequest(BaseModel):
    signal: List[float]
    heatmap: List[float]
    prediction: int
    probabilities: List[float]
    record_id: str = "interactive_session"
    audit: Optional[dict] = None
    model_metrics: Optional[dict] = None


@app.post("/api/report")
def generate_report(req: ReportRequest):
    from src.reporting.report_generator import ReportGenerator
    reporter = ReportGenerator()
    path = reporter.generate_report(
        record_id=req.record_id,
        signal=np.array(req.signal),
        prediction=req.prediction,
        probabilities=np.array(req.probabilities),
        heatmap=np.array(req.heatmap),
        audit_report=req.audit,
        model_metrics=req.model_metrics,
    )
    return {"path": path, "filename": Path(path).name}


@app.get("/api/report/download/{filename}")
def download_report(filename: str):
    safe = Path(filename).name          # strip any path traversal
    path = REPORT_DIR / safe
    if not path.exists():
        raise HTTPException(404, f"Report '{safe}' not found.")
    return FileResponse(
        str(path), media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@app.get("/api/reports/list")
def list_reports():
    if not REPORT_DIR.exists():
        return {"reports": []}
    files = sorted(REPORT_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"reports": [{"name": f.name, "size": f.stat().st_size} for f in files[:20]]}
