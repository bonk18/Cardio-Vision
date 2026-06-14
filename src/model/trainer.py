"""
CardioVision — Training Pipeline
Dataset, Trainer, and evaluation utilities with CUDA acceleration.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from typing import Tuple, Optional, Dict
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, NUM_CLASSES, CLASS_NAMES,
    MODEL_DIR, NUM_WORKERS, PIN_MEMORY,
)
from src.utils.helpers import set_seed, format_duration


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG beat segments."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, W)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    """Training pipeline with balanced sampling, LR scheduling, early stopping."""

    def __init__(self, model, device=DEVICE, learning_rate=LEARNING_RATE,
                 weight_decay=WEIGHT_DECAY, num_epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE, patience=EARLY_STOPPING_PATIENCE,
                 save_dir=None):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR,
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [], "lr": [],
        }
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def prepare_data(self, X, y, seed=42):
        """Split data and create DataLoaders with balanced sampling."""
        set_seed(seed)
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=seed, stratify=y)
        val_r = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_r, random_state=seed, stratify=y_tv)

        print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name}: train={np.sum(y_train==i)}, val={np.sum(y_val==i)}, test={np.sum(y_test==i)}")

        train_ds = ECGDataset(X_train, y_train)
        val_ds = ECGDataset(X_val, y_val)
        test_ds = ECGDataset(X_test, y_test)

        counts = np.bincount(y_train, minlength=NUM_CLASSES)
        weights = 1.0 / (np.sqrt(counts) + 1e-6)
        sampler = WeightedRandomSampler(weights[y_train], len(y_train), replacement=True)

        kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, sampler=sampler, drop_last=True, **kw)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, **kw)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, **kw)
        return train_loader, val_loader, test_loader

    def train_epoch(self, loader):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device, non_blocking=True), y_b.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(X_b)
            loss = self.criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * X_b.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
        n = len(all_labels)
        return total_loss/n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device, non_blocking=True), y_b.to(self.device, non_blocking=True)
            logits = self.model(X_b)
            loss = self.criterion(logits, y_b)
            total_loss += loss.item() * X_b.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
        n = len(all_labels)
        return total_loss/n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro", zero_division=0)

    def train(self, train_loader, val_loader, save_filename="best_model.pt"):
        """Full training loop with early stopping and checkpointing."""
        print(f"\n{'='*60}")
        print(f"  Training on {self.device} for up to {self.num_epochs} epochs")
        print(f"  Batch size: {self.batch_size} | LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")
        start = time.time()

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            tl, ta, tf = self.train_epoch(train_loader)
            vl, va, vf = self.eval_epoch(val_loader)
            self.scheduler.step(vl)
            lr = self.optimizer.param_groups[0]["lr"]

            for k, v in zip(self.history.keys(), [tl, vl, ta, va, tf, vf, lr]):
                self.history[k].append(v)

            print(f"  Epoch {epoch:3d}/{self.num_epochs} | "
                  f"Loss: {tl:.4f}/{vl:.4f} | Acc: {ta:.3f}/{va:.3f} | "
                  f"F1: {tf:.3f}/{vf:.3f} | LR: {lr:.2e} | {time.time()-t0:.1f}s")

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.epochs_no_improve = 0
                self.save_checkpoint(save_filename, epoch, vl, va)
                print(f"             -> Best model saved to {save_filename} (val_loss={vl:.4f})")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        print(f"\n  Training complete in {format_duration(time.time()-start)}")
        return self.history

    def save_checkpoint(self, filename, epoch, val_loss, val_acc):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss, "val_acc": val_acc, "history": self.history,
        }, self.save_dir / filename)

    def load_checkpoint(self, filename="best_model.pt"):
        ckpt = torch.load(self.save_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)
        print(f"  Loaded checkpoint epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    @torch.no_grad()
    def evaluate(self, test_loader):
        """Full evaluation on test set."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for X_b, y_b in test_loader:
            X_b = X_b.to(self.device, non_blocking=True)
            logits = self.model(X_b)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.numpy())
            all_probs.extend(probs.cpu().numpy())

        preds, labels, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
        report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        print(f"\n{'='*60}\n  TEST SET EVALUATION\n{'='*60}")
        print(f"  Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

        return {"accuracy": acc, "macro_f1": f1, "classification_report": report,
                "confusion_matrix": cm.tolist(), "predictions": preds,
                "labels": labels, "probabilities": probs}
