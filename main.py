"""
CardioVision — CLI Entry Point
Download data, preprocess, train, evaluate, and generate reports from the command line.
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import DEVICE, MODEL_DIR, CLASS_NAMES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.utils.helpers import set_seed, get_device_info, count_parameters

def get_model(arch, device=DEVICE):
    if arch == "hybrid":
        from src.model.hybrid_model import HybridCNNLSTM
        return HybridCNNLSTM().to(device)
    elif arch == "attention":
        from src.model.attention_model import AttentionHybridModel
        return AttentionHybridModel().to(device)
    elif arch == "resnet":
        from src.model.resnet_model import ResNetHybridModel
        return ResNetHybridModel().to(device)
    elif arch == "se":
        from src.model.se_model import SEHybridModel
        return SEHybridModel().to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")



def cmd_download(args):
    """Download MIT-BIH database."""
    from src.preprocessing.loader import ECGLoader
    print("\n📥 Downloading MIT-BIH Arrhythmia Database...")
    loader = ECGLoader()
    records = args.records.split(",") if args.records else None
    loader.download_mitbih(records=records)
    print("  ✓ Download complete.\n")


def cmd_preprocess(args):
    """Load and preprocess MIT-BIH data."""
    from src.preprocessing.loader import ECGLoader
    print("\n🔬 Loading & preprocessing MIT-BIH data...")
    loader = ECGLoader()
    X, y, record_ids, reports = loader.load_all_mitbih()
    print(f"  ✓ Loaded {X.shape[0]} beats ({X.shape[1]} samples each)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {np.sum(y == i)}")

    # Print audit summary
    n_warn = sum(1 for r in reports if r.severity == "WARNING")
    n_crit = sum(1 for r in reports if r.severity == "CRITICAL")
    print(f"\n  Audit: {len(reports)} records | {n_warn} warnings | {n_crit} critical")

    # Save processed data
    from config import PROCESSED_DATA_DIR
    np.savez_compressed(PROCESSED_DATA_DIR / "mitbih_processed.npz", X=X, y=y, record_ids=record_ids)
    print(f"  ✓ Saved to {PROCESSED_DATA_DIR / 'mitbih_processed.npz'}\n")
    return X, y


def cmd_train(args):
    """Train the model."""
    from config import PROCESSED_DATA_DIR
    from src.model.trainer import Trainer

    set_seed(args.seed)
    print(f"\n🧠 Training CardioVision Model")
    print(f"  Device: {DEVICE}")
    info = get_device_info()
    if info["cuda_available"]:
        print(f"  GPU: {info['device_name']} ({info['total_memory_gb']} GB)")

    # Load data
    data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
    if not data_path.exists():
        print("  ⚠ Processed data not found. Running preprocessing first...")
        cmd_preprocess(args)
    data = np.load(data_path)
    X, y = data["X"], data["y"]

    # Build model
    model = get_model(args.arch, DEVICE)
    print(f"  Architecture: {args.arch}")
    print(f"  Parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(
        model, device=DEVICE, num_epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
    )
    train_loader, val_loader, test_loader = trainer.prepare_data(X, y, seed=args.seed)
    
    save_filename = f"best_model_{args.arch}.pt"
    trainer.train(train_loader, val_loader, save_filename=save_filename)

    # Evaluate
    results = trainer.evaluate(test_loader)
    print(f"\n  ✓ Model saved to {MODEL_DIR / save_filename}\n")
    return results


def cmd_evaluate(args):
    """Evaluate a saved model on test data."""
    from config import PROCESSED_DATA_DIR
    from src.model.trainer import Trainer

    data = np.load(PROCESSED_DATA_DIR / "mitbih_processed.npz")
    X, y = data["X"], data["y"]

    model = get_model(args.arch, DEVICE)
    trainer = Trainer(model, device=DEVICE)
    trainer.load_checkpoint(f"best_model_{args.arch}.pt")
    _, _, test_loader = trainer.prepare_data(X, y)
    trainer.evaluate(test_loader)


def cmd_report(args):
    """Generate a sample clinical report."""
    from src.model.trainer import Trainer
    from src.explainability.gradcam import GradCAM1D
    from src.reporting.report_generator import ReportGenerator
    from config import PROCESSED_DATA_DIR

    data = np.load(PROCESSED_DATA_DIR / "mitbih_processed.npz")
    X, y = data["X"], data["y"]

    model = get_model(args.arch, DEVICE)
    trainer = Trainer(model, device=DEVICE)
    trainer.load_checkpoint(f"best_model_{args.arch}.pt")

    gradcam = GradCAM1D(model)
    reporter = ReportGenerator()

    # Generate report for a random sample from each class
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y == cls_idx
        if not np.any(mask):
            continue
        idx = np.where(mask)[0][0]
        sample = torch.FloatTensor(X[idx]).unsqueeze(0).unsqueeze(0)
        heatmap, pred, probs = gradcam.compute(sample)
        path = reporter.generate_report(
            record_id=f"sample_{cls_name}",
            signal=X[idx], prediction=pred,
            probabilities=probs, heatmap=heatmap,
        )
        print(f"  ✓ Report for {cls_name}: {path}")

    gradcam.remove_hooks()


def cmd_test(args):
    """Run unit tests."""
    import pytest
    pytest.main(["tests/", "-v", "--tb=short"])


def main():
    parser = argparse.ArgumentParser(
        description="CardioVision — Cardiac Arrhythmia Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Download
    p_dl = sub.add_parser("download", help="Download MIT-BIH database")
    p_dl.add_argument("--records", type=str, default=None, help="Comma-separated record IDs")

    # Preprocess
    sub.add_parser("preprocess", help="Preprocess MIT-BIH data")

    # Train
    p_tr = sub.add_parser("train", help="Train the model")
    p_tr.add_argument("--arch", type=str, default="hybrid", choices=["hybrid", "attention", "resnet", "se"])
    p_tr.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p_tr.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_tr.add_argument("--lr", type=float, default=LEARNING_RATE)
    p_tr.add_argument("--seed", type=int, default=42)

    # Evaluate
    p_ev = sub.add_parser("evaluate", help="Evaluate saved model")
    p_ev.add_argument("--arch", type=str, default="hybrid", choices=["hybrid", "attention", "resnet", "se"])

    # Report
    p_rep = sub.add_parser("report", help="Generate clinical reports")
    p_rep.add_argument("--arch", type=str, default="hybrid", choices=["hybrid", "attention", "resnet", "se"])

    # Test
    sub.add_parser("test", help="Run unit tests")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()