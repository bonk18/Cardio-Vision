"""
CardioVision — Clinical Audit Report Generator
Generates HTML reports with signal analysis, predictions, and Grad-CAM overlays.
"""

import os
import json
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CLASS_NAMES, REPORT_DIR


class ReportGenerator:
    """
    Generates comprehensive clinical audit reports in HTML format.

    Reports include:
    - Signal quality audit summary
    - Preprocessing steps applied
    - Model prediction with confidence scores
    - Grad-CAM heatmap visualization
    - Exportable cleaned dataset
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else REPORT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        record_id: str,
        signal: np.ndarray,
        prediction: int,
        probabilities: np.ndarray,
        heatmap: np.ndarray,
        audit_report: Optional[Dict] = None,
        model_metrics: Optional[Dict] = None,
    ) -> str:
        """
        Generate a full HTML clinical audit report.

        Returns the file path of the saved report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pred_class = CLASS_NAMES[prediction]
        confidence = float(probabilities[prediction]) * 100

        # Generate signal + heatmap plot as base64
        plot_b64 = self._generate_plot_base64(signal, heatmap, pred_class)

        # Build probability bar data
        prob_bars = ""
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            pct = float(prob) * 100
            color = self._get_class_color(i)
            is_pred = "★ " if i == prediction else ""
            prob_bars += f"""
            <div class="prob-row">
                <span class="prob-label">{is_pred}{name}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
                <span class="prob-value">{pct:.1f}%</span>
            </div>"""

        # Audit section
        audit_html = self._render_audit(audit_report) if audit_report else "<p>No audit data available.</p>"

        # Model metrics section
        metrics_html = self._render_metrics(model_metrics) if model_metrics else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioVision Report — {record_id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#0d1117; color:#c9d1d9; line-height:1.6; padding:24px; }}
  .container {{ max-width:900px; margin:0 auto; }}
  .header {{ background:linear-gradient(135deg,#1a1b2e,#16213e); border-radius:16px; padding:32px;
             margin-bottom:24px; border:1px solid #30363d; }}
  .header h1 {{ color:#58a6ff; font-size:28px; font-weight:700; margin-bottom:4px; }}
  .header .subtitle {{ color:#8b949e; font-size:14px; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-top:12px; }}
  .badge-normal {{ background:#1a4731; color:#3fb950; }}
  .badge-afib {{ background:#4a2000; color:#f0883e; }}
  .badge-pvc {{ background:#4a0e0e; color:#f85149; }}
  .badge-global {{ background:#1c2541; color:#79c0ff; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:24px; margin-bottom:20px; }}
  .card h2 {{ color:#58a6ff; font-size:18px; margin-bottom:16px; border-bottom:1px solid #21262d; padding-bottom:8px; }}
  .plot-container {{ text-align:center; }}
  .plot-container img {{ max-width:100%; border-radius:8px; }}
  .prob-row {{ display:flex; align-items:center; margin-bottom:8px; gap:12px; }}
  .prob-label {{ width:100px; font-size:13px; font-weight:500; text-align:right; }}
  .prob-bar-bg {{ flex:1; height:24px; background:#21262d; border-radius:6px; overflow:hidden; }}
  .prob-bar-fill {{ height:100%; border-radius:6px; transition:width 0.5s; }}
  .prob-value {{ width:60px; font-size:13px; font-weight:600; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
  .stat-item {{ background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:12px; text-align:center; }}
  .stat-item .value {{ font-size:24px; font-weight:700; color:#58a6ff; }}
  .stat-item .label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; }}
  .severity-pass {{ color:#3fb950; }} .severity-warning {{ color:#f0883e; }} .severity-critical {{ color:#f85149; }}
  .issue-list {{ list-style:none; }}
  .issue-list li {{ padding:6px 0; border-bottom:1px solid #21262d; font-size:13px; }}
  .issue-list li:before {{ content:"⚠ "; color:#f0883e; }}
  .footer {{ text-align:center; padding:20px; color:#484f58; font-size:12px; }}
  .confidence-big {{ font-size:48px; font-weight:700; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>CardioVision — Clinical Audit Report</h1>
    <div class="subtitle">Record: {record_id} | Generated: {timestamp}</div>
    <span class="badge badge-{pred_class.lower()}">{pred_class} Detected</span>
  </div>

  <div class="card">
    <h2>Classification Result</h2>
    <div style="display:flex;align-items:center;gap:32px;margin-bottom:20px">
      <div style="text-align:center">
        <div class="confidence-big" style="color:{self._get_class_color(prediction)}">{confidence:.1f}%</div>
        <div style="color:#8b949e;font-size:12px">Confidence</div>
      </div>
      <div style="flex:1">{prob_bars}</div>
    </div>
  </div>

  <div class="card">
    <h2>ECG Signal with Grad-CAM Heatmap</h2>
    <div class="plot-container">
      <img src="data:image/png;base64,{plot_b64}" alt="ECG with Grad-CAM overlay"/>
    </div>
    <p style="color:#8b949e;font-size:12px;margin-top:8px;text-align:center">
      Red-highlighted regions indicate areas most influential in the model's decision.
    </p>
  </div>

  <div class="card">
    <h2>Signal Quality Audit</h2>
    {audit_html}
  </div>

  {metrics_html}

  <div class="footer">
    <p>CardioVision — ML-Explainable Hybrid CNN-LSTM System for Cardiac Arrhythmia Detection</p>
    <p>This report is generated by an AI system and should be reviewed by a qualified clinician.</p>
  </div>
</div>
</body></html>"""

        # Save report
        filename = f"report_{record_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        filepath.write_text(html, encoding="utf-8")
        return str(filepath)

    def _generate_plot_base64(self, signal, heatmap, pred_class):
        """Generate matplotlib plot and return as base64 PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#161b22")
        ax.set_facecolor("#0d1117")

        x = np.arange(len(signal))

        # Color the ECG line by heatmap intensity
        points = np.array([x, signal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="YlOrRd", linewidth=1.5)
        lc.set_array(heatmap[:-1])
        lc.set_clim(0, 1)
        ax.add_collection(lc)

        # Fill under high-attention regions
        threshold = 0.4
        high_att = heatmap > threshold
        if np.any(high_att):
            ax.fill_between(x, signal.min() - 0.3, signal.max() + 0.3,
                          where=high_att, alpha=0.15, color="#f85149")

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(signal.min() - 0.5, signal.max() + 0.5)
        ax.set_xlabel("Sample", color="#8b949e", fontsize=10)
        ax.set_ylabel("Amplitude (normalized)", color="#8b949e", fontsize=10)
        ax.set_title(f"ECG Beat — Prediction: {pred_class}", color="#c9d1d9", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#484f58")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        cbar = plt.colorbar(lc, ax=ax, pad=0.02)
        cbar.set_label("Grad-CAM Attention", color="#8b949e", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#484f58")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#484f58")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _get_class_color(self, class_idx):
        colors = ["#3fb950", "#f0883e", "#f85149", "#79c0ff"]
        return colors[class_idx % len(colors)]

    def _render_audit(self, audit):
        if isinstance(audit, dict):
            d = audit
        elif hasattr(audit, "to_dict"):
            d = audit.to_dict()
        else:
            return "<p>Invalid audit data.</p>"

        severity = d.get("severity", "PASS")
        sev_class = f"severity-{severity.lower()}"
        issues = d.get("issues", [])

        issue_html = ""
        if issues:
            items = "".join(f"<li>{iss}</li>" for iss in issues)
            issue_html = f'<ul class="issue-list">{items}</ul>'
        else:
            issue_html = '<p style="color:#3fb950">✓ No issues detected — signal quality is good.</p>'

        return f"""
        <div class="stat-grid">
          <div class="stat-item"><div class="value">{d.get('total_samples', 'N/A')}</div><div class="label">Total Samples</div></div>
          <div class="stat-item"><div class="value">{d.get('duration_sec', 'N/A')}</div><div class="label">Duration (s)</div></div>
          <div class="stat-item"><div class="value">{d.get('missing_pct', 0):.1f}%</div><div class="label">Missing Data</div></div>
          <div class="stat-item"><div class="value {sev_class}">{severity}</div><div class="label">Severity</div></div>
        </div>
        <div style="margin-top:16px">{issue_html}</div>"""

    def _render_metrics(self, metrics):
        if not metrics:
            return ""
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("macro_f1", 0)
        return f"""
        <div class="card">
          <h2>Model Performance (Test Set)</h2>
          <div class="stat-grid">
            <div class="stat-item"><div class="value">{acc:.1%}</div><div class="label">Accuracy</div></div>
            <div class="stat-item"><div class="value">{f1:.1%}</div><div class="label">Macro F1</div></div>
          </div>
        </div>"""

    def export_cleaned_dataset(self, X, y, filename="cleaned_dataset.npz"):
        """Export the cleaned, preprocessed dataset."""
        path = self.output_dir / filename
        np.savez_compressed(path, X=X, y=y, class_names=CLASS_NAMES)
        return str(path)
