"""
CardioVision — Streamlit Interactive Dashboard
Premium dark-themed interface for ECG analysis, classification, and reporting.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os, sys, io, time, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, CLASS_NAMES, NUM_CLASSES, BEAT_WINDOW, SAMPLING_RATE,
    MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
)

# ──────────────────────── Page Config ────────────────────────
st.set_page_config(
    page_title="CardioVision — Arrhythmia Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────── Custom CSS ────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-primary: #0a0e17; --bg-secondary: #111827; --bg-card: rgba(17,24,39,0.7);
    --border: rgba(99,102,241,0.15); --text-primary: #e2e8f0; --text-secondary: #94a3b8;
    --accent: #6366f1; --accent-glow: rgba(99,102,241,0.3);
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444; --info: #3b82f6;
}
.stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #c7d2fe !important; }
.glass-card {
    background: var(--bg-card); backdrop-filter: blur(20px);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 24px; margin-bottom: 16px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.05));
    border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; text-align: center;
}
.metric-value { font-size: 32px; font-weight: 800; color: var(--accent); }
.metric-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1.5px; }
.hero-title {
    font-size: 36px; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub { color: var(--text-secondary); font-size: 14px; margin-bottom: 24px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-normal { background: rgba(16,185,129,0.15); color: #10b981; }
.badge-afib { background: rgba(245,158,11,0.15); color: #f59e0b; }
.badge-pvc { background: rgba(239,68,68,0.15); color: #ef4444; }
.badge-global { background: rgba(59,130,246,0.15); color: #3b82f6; }
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 8px 24px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
}
</style>""", unsafe_allow_html=True)


# ──────────────────────── Sidebar ────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🫀 CardioVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML-Explainable Cardiac Arrhythmia Detection</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📤 Upload & Preprocess",
        "🧠 Train Model",
        "🔍 Inference & Explain",
        "📊 Reports",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("### ⚙️ Settings")
    selected_arch = st.selectbox("Architecture", ["hybrid", "attention", "resnet", "se"])

    st.divider()
    st.caption(f"Device: **{'🟢 CUDA' if torch.cuda.is_available() else '🟡 CPU'}**")
    if torch.cuda.is_available():
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────── Helpers ────────────────────────
CLASS_COLORS = {"Normal": "#10b981", "AFib": "#f59e0b", "PVC": "#ef4444", "Global": "#3b82f6"}

def make_ecg_plot(signal, title="ECG Signal", heatmap=None, fs=SAMPLING_RATE):
    """Create a Plotly ECG plot with optional Grad-CAM overlay."""
    t = np.arange(len(signal)) / fs
    fig = go.Figure()
    if heatmap is not None:
        # Color segments by heatmap
        for i in range(len(signal) - 1):
            r = int(255 * heatmap[i])
            g = int(100 * (1 - heatmap[i]))
            b = int(200 * (1 - heatmap[i]))
            fig.add_trace(go.Scatter(
                x=t[i:i+2], y=signal[i:i+2], mode='lines',
                line=dict(color=f'rgb({r},{g},{b})', width=2),
                showlegend=False, hoverinfo='skip',
            ))
        # Add heatmap fill
        high = heatmap > 0.4
        if np.any(high):
            fig.add_trace(go.Scatter(
                x=t, y=np.where(high, signal.max() + 0.3, np.nan),
                fill='tozeroy', fillcolor='rgba(239,68,68,0.08)',
                line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))
    else:
        fig.add_trace(go.Scatter(x=t, y=signal, mode='lines',
                                 line=dict(color='#6366f1', width=1.5), name='ECG'))
    fig.update_layout(
        template='plotly_dark', title=title,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.5)',
        xaxis_title='Time (s)', yaxis_title='Amplitude',
        height=350, margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family='Inter', color='#94a3b8'),
    )
    return fig


# ──────────────────────── Pages ────────────────────────

# ═══════════════ DASHBOARD ═══════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="hero-title">CardioVision Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Hybrid CNN-LSTM System for Cardiac Arrhythmia Detection with Explainable AI</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    model_exists = (MODEL_DIR / f"best_model_{selected_arch}.pt").exists()
    # Also check legacy model name if specific one not found
    if not model_exists and selected_arch == "hybrid":
        model_exists = (MODEL_DIR / "best_model.pt").exists()
        
    data_exists = (PROCESSED_DATA_DIR / "mitbih_processed.npz").exists()

    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Arrhythmia Classes</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{"✓" if data_exists else "—"}</div><div class="metric-label">Data Ready</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{"✓" if model_exists else "—"}</div><div class="metric-label">Model Trained</div></div>', unsafe_allow_html=True)
    with c4:
        dev = "CUDA" if torch.cuda.is_available() else "CPU"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{dev}</div><div class="metric-label">Compute Device</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        ```
        Input (1, 360)
          → Conv1D(32) → BN → ReLU → Pool
          → Conv1D(64) → BN → ReLU → Pool
          → Conv1D(128) → BN → ReLU → Pool  ← Grad-CAM
          → BiLSTM(64×2)
          → FC(64) → Softmax(4)
        ```
        """)
        st.markdown("**Classes:** Normal (Healthy) · AFib (Atrial/Supraventricular) · PVC (Ventricular) · Global (Pacemaker/Fusion/Other)")

    with col2:
        st.markdown("### 📋 Pipeline")
        st.markdown("""
        1. **Upload** ECG (MIT-BIH / CSV / EDF)
        2. **Audit** — detect missing data, duplicates, noise
        3. **Denoise** — Butterworth bandpass + Notch filter
        4. **Segment** — R-peak centered beats
        5. **Classify** — Hybrid CNN-LSTM inference
        6. **Explain** — Grad-CAM heatmap overlay
        7. **Report** — downloadable HTML clinical report
        """)


# ═══════════════ UPLOAD & PREPROCESS ═══════════════
elif page == "📤 Upload & Preprocess":
    st.markdown('<div class="hero-title">Upload & Preprocess</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Upload File", "🗂️ MIT-BIH Database"])

    with tab1:
        uploaded = st.file_uploader("Upload ECG signal (CSV or EDF)", type=["csv", "edf"])
        if uploaded:
            from src.preprocessing.loader import ECGLoader
            loader = ECGLoader()
            with st.spinner("Processing..."):
                if uploaded.name.endswith(".csv"):
                    tmp = Path(tempfile.mktemp(suffix=".csv"))
                    tmp.write_bytes(uploaded.read())
                    data = loader.load_csv(str(tmp))
                    tmp.unlink()
                elif uploaded.name.endswith(".edf"):
                    tmp = Path(tempfile.mktemp(suffix=".edf"))
                    tmp.write_bytes(uploaded.read())
                    data = loader.load_edf(str(tmp))
                    tmp.unlink()

            signal = data["signal"]
            audit = data["audit_report"]
            st.session_state["uploaded_signal"] = signal
            st.session_state["uploaded_fs"] = data["fs"]
            st.session_state["uploaded_audit"] = audit

            st.plotly_chart(make_ecg_plot(signal[:3600], f"Uploaded: {uploaded.name}"), use_container_width=True)

            # Audit results
            st.markdown("### 🔍 Audit Results")
            ad = audit.to_dict()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Samples", f"{ad['total_samples']:,}")
            c2.metric("Duration", f"{ad['duration_sec']:.1f}s")
            c3.metric("Missing", f"{ad['missing_pct']:.1f}%")
            sev = ad['severity']
            c4.metric("Severity", sev)

            if ad["issues"]:
                for iss in ad["issues"]:
                    st.warning(iss)
            else:
                st.success("✓ Signal quality is good — no issues detected.")

    with tab2:
        st.markdown("Download and preprocess the MIT-BIH Arrhythmia Database.")
        if st.button("⬇️ Download & Process MIT-BIH", type="primary"):
            from src.preprocessing.loader import ECGLoader
            loader = ECGLoader()
            with st.status("Processing MIT-BIH...", expanded=True) as status:
                st.write("Downloading records...")
                loader.download_mitbih()
                st.write("Segmenting beats...")
                X, y, record_ids, reports = loader.load_all_mitbih()
                np.savez_compressed(PROCESSED_DATA_DIR / "mitbih_processed.npz", X=X, y=y, record_ids=record_ids)
                status.update(label=f"✓ Done — {X.shape[0]:,} beats extracted", state="complete")

            st.session_state["X"] = X
            st.session_state["y"] = y

            # Distribution chart
            counts = [int(np.sum(y == i)) for i in range(NUM_CLASSES)]
            fig = go.Figure(go.Bar(
                x=CLASS_NAMES, y=counts,
                marker_color=[CLASS_COLORS[n] for n in CLASS_NAMES],
                text=counts, textposition='auto',
            ))
            fig.update_layout(template='plotly_dark', title='Class Distribution',
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.5)',
                              height=350, font=dict(family='Inter', color='#94a3b8'))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════ TRAIN MODEL ═══════════════
elif page == "🧠 Train Model":
    st.markdown('<div class="hero-title">Train Model</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", 5, 200, 50)
    batch_size = col2.number_input("Batch Size", 32, 512, 128, step=32)
    lr = col3.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")

    if st.button("🚀 Start Training", type="primary"):
        data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
        if not data_path.exists():
            st.error("⚠ No processed data. Go to Upload & Preprocess first.")
        else:
            from main import get_model
            from src.model.trainer import Trainer
            from src.utils.helpers import set_seed, count_parameters

            set_seed(42)
            data = np.load(data_path)
            X, y = data["X"], data["y"]

            model = get_model(selected_arch, DEVICE)
            st.info(f"Model parameters: **{count_parameters(model):,}** | Device: **{DEVICE}**")

            trainer = Trainer(model, device=DEVICE, num_epochs=epochs,
                              batch_size=batch_size, learning_rate=lr)
            train_ld, val_ld, test_ld = trainer.prepare_data(X, y)

            progress = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()

            # Training loop with live updates
            train_losses, val_losses, train_accs, val_accs = [], [], [], []
            model.train()
            for ep in range(1, epochs + 1):
                tl, ta, tf = trainer.train_epoch(train_ld)
                vl, va, vf = trainer.eval_epoch(val_ld)
                trainer.scheduler.step(vl)

                for k, v in zip(trainer.history.keys(),
                                [tl, vl, ta, va, tf, vf, trainer.optimizer.param_groups[0]["lr"]]):
                    trainer.history[k].append(v)

                if vl < trainer.best_val_loss:
                    trainer.best_val_loss = vl
                    trainer.epochs_no_improve = 0
                    trainer.save_checkpoint(f"best_model_{selected_arch}.pt", ep, vl, va)
                else:
                    trainer.epochs_no_improve += 1

                train_losses.append(tl); val_losses.append(vl)
                train_accs.append(ta); val_accs.append(va)
                progress.progress(ep / epochs)
                status_text.markdown(f"**Epoch {ep}/{epochs}** — Loss: {tl:.4f}/{vl:.4f} | Acc: {ta:.3f}/{va:.3f}")

                if ep % 5 == 0 or ep == epochs:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                    fig.add_trace(go.Scatter(y=train_losses, name='Train', line=dict(color='#6366f1')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=val_losses, name='Val', line=dict(color='#a78bfa')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=train_accs, name='Train', line=dict(color='#10b981'), showlegend=False), row=1, col=2)
                    fig.add_trace(go.Scatter(y=val_accs, name='Val', line=dict(color='#34d399'), showlegend=False), row=1, col=2)
                    fig.update_layout(template='plotly_dark', height=300,
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.5)',
                                      font=dict(family='Inter', color='#94a3b8'))
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                if trainer.epochs_no_improve >= trainer.patience:
                    status_text.markdown(f"**⚡ Early stopping at epoch {ep}**")
                    break

            # Test evaluation
            trainer.load_checkpoint(f"best_model_{selected_arch}.pt")
            results = trainer.evaluate(test_loader)
            st.session_state["test_results"] = results

            st.success(f"✓ Training complete — Test Accuracy: **{results['accuracy']:.1%}** | Macro F1: **{results['macro_f1']:.1%}**")

            # Confusion matrix
            cm = np.array(results["confusion_matrix"])
            fig_cm = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                               color_continuous_scale="Viridis", title="Confusion Matrix")
            fig_cm.update_layout(template='plotly_dark', height=400,
                                 paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94a3b8'))
            st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════ INFERENCE & EXPLAIN ═══════════════
elif page == "🔍 Inference & Explain":
    st.markdown('<div class="hero-title">Inference & Explainability</div>', unsafe_allow_html=True)

    model_path = MODEL_DIR / f"best_model_{selected_arch}.pt"
    if not model_path.exists() and selected_arch == "hybrid":
        model_path = MODEL_DIR / "best_model.pt"
        
    if not model_path.exists():
        st.warning(f"⚠ No trained model found for {selected_arch}. Train the model first.")
    else:
        from main import get_model
        from src.model.trainer import Trainer
        from src.explainability.gradcam import GradCAM1D

        @st.cache_resource
        def load_model(arch, path):
            model = get_model(arch, DEVICE)
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

        model = load_model(selected_arch, model_path)
        gradcam = GradCAM1D(model)

        source = st.radio("Signal source", ["📁 From processed dataset", "📤 From uploaded file"], horizontal=True)

        sample_tensor = None
        raw_signal = None

        if source == "📁 From processed dataset":
            data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
            if data_path.exists():
                data = np.load(data_path)
                X, y = data["X"], data["y"]
                record_ids = data.get("record_ids", None)
                
                cls = st.selectbox("Filter by class", ["Any"] + CLASS_NAMES)
                if cls != "Any":
                    mask = y == CLASS_NAMES.index(cls)
                    X_filt, y_filt = X[mask], y[mask]
                    if record_ids is not None:
                        rec_filt = record_ids[mask]
                else:
                    X_filt, y_filt = X, y
                    if record_ids is not None:
                        rec_filt = record_ids
                        
                idx = st.slider("Sample index", 0, len(X_filt) - 1, 0)
                raw_signal = X_filt[idx]
                sample_tensor = torch.FloatTensor(raw_signal).unsqueeze(0).unsqueeze(0)
                
                patient_info = f" | Patient/Record ID: **{rec_filt[idx]}**" if record_ids is not None else ""
                st.caption(f"True label: **{CLASS_NAMES[y_filt[idx]]}**{patient_info}")
            else:
                st.error("No processed data available.")

        elif source == "📤 From uploaded file":
            if "uploaded_signal" in st.session_state:
                from src.preprocessing.loader import ECGLoader
                loader = ECGLoader(apply_filters=False, apply_imputation=False)
                sig = st.session_state["uploaded_signal"]
                fs = st.session_state.get("uploaded_fs", SAMPLING_RATE)
                segments = loader.segment_uploaded_signal(sig, fs=fs)
                if len(segments) > 0:
                    idx = st.slider("Beat index", 0, len(segments) - 1, 0)
                    raw_signal = segments[idx]
                    sample_tensor = torch.FloatTensor(raw_signal).unsqueeze(0).unsqueeze(0)
                else:
                    st.error("No beats segmented from uploaded signal.")
            else:
                st.info("Upload a file in the Upload & Preprocess page first.")

        if sample_tensor is not None and st.button("🔬 Run Inference + Grad-CAM", type="primary"):
            with st.spinner("Computing..."):
                heatmap, pred, probs = gradcam.compute(sample_tensor)

            st.session_state["last_inference"] = {
                "signal": raw_signal, "heatmap": heatmap,
                "prediction": pred, "probabilities": probs,
            }

            # Results
            pred_name = CLASS_NAMES[pred]
            conf = probs[pred] * 100
            badge_cls = pred_name.lower()
            st.markdown(f'### Prediction: <span class="badge badge-{badge_cls}">{pred_name}</span> ({conf:.1f}% confidence)', unsafe_allow_html=True)

            # Probability bars
            CLASS_DESCRIPTIONS = {
                "Normal": "Normal (Healthy)",
                "AFib": "AFib (Atrial)",
                "PVC": "PVC (Ventricular)",
                "Global": "Global (Paced/Fusion)"
            }
            cols = st.columns(NUM_CLASSES)
            for i, (col, name) in enumerate(zip(cols, CLASS_NAMES)):
                col.metric(CLASS_DESCRIPTIONS.get(name, name), f"{probs[i]*100:.1f}%")

            # ECG + Grad-CAM
            st.plotly_chart(
                make_ecg_plot(raw_signal, f"Grad-CAM Overlay — {pred_name}", heatmap),
                use_container_width=True,
            )

            # Raw ECG for comparison
            st.plotly_chart(
                make_ecg_plot(raw_signal, "Original ECG Signal"),
                use_container_width=True,
            )


# ═══════════════ REPORTS ═══════════════
elif page == "📊 Reports":
    st.markdown('<div class="hero-title">Clinical Reports</div>', unsafe_allow_html=True)

    if "last_inference" not in st.session_state:
        st.info("Run inference on a sample first (Inference & Explain page).")
    else:
        inf = st.session_state["last_inference"]
        st.markdown(f"**Last prediction:** {CLASS_NAMES[inf['prediction']]} ({inf['probabilities'][inf['prediction']]*100:.1f}%)")

        if st.button("📄 Generate Clinical Report", type="primary"):
            from src.reporting.report_generator import ReportGenerator
            reporter = ReportGenerator()
            audit = st.session_state.get("uploaded_audit", None)
            metrics = st.session_state.get("test_results", None)

            path = reporter.generate_report(
                record_id="interactive_session",
                signal=inf["signal"],
                prediction=inf["prediction"],
                probabilities=inf["probabilities"],
                heatmap=inf["heatmap"],
                audit_report=audit,
                model_metrics=metrics,
            )
            st.success(f"✓ Report saved to `{path}`")

            with open(path, "r") as f:
                html = f.read()
            st.download_button("⬇️ Download HTML Report", html,
                               file_name="cardiovision_report.html", mime="text/html")
            st.components.v1.html(html, height=800, scrolling=True)

    # Export cleaned dataset
    st.markdown("---")
    st.markdown("### 📦 Export Cleaned Dataset")
    data_path = PROCESSED_DATA_DIR / "mitbih_processed.npz"
    if data_path.exists():
        data = np.load(data_path)
        st.markdown(f"Dataset: **{data['X'].shape[0]:,}** beats × **{data['X'].shape[1]}** samples")
        buf = io.BytesIO()
        np.savez_compressed(buf, X=data["X"], y=data["y"], class_names=CLASS_NAMES)
        st.download_button("⬇️ Download Cleaned Dataset (.npz)", buf.getvalue(),
                           file_name="cardiovision_cleaned_dataset.npz")
    else:
        st.info("Process data first to enable export.")
