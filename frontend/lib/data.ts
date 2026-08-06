export const ARRHYTHMIA_CLASSES = [
  {
    id: 0,
    name: "Normal",
    short: "NSR",
    color: "#10b981",
    bgColor: "rgba(16,185,129,0.1)",
    borderColor: "rgba(16,185,129,0.3)",
    annotations: ["N", "L", "R"],
    description:
      "Normal sinus rhythm. Includes left and right bundle branch block beats that are morphologically normal variants.",
    clinicalNote: "These beats represent healthy cardiac conduction through the normal His-Purkinje pathway.",
    prevalence: "~70%",
    icon: "💚",
  },
  {
    id: 1,
    name: "AFib",
    short: "SVT",
    color: "#f59e0b",
    bgColor: "rgba(245,158,11,0.1)",
    borderColor: "rgba(245,158,11,0.3)",
    annotations: ["A", "a", "J", "S", "e", "j"],
    description:
      "Atrial fibrillation and supraventricular ectopic beats. Includes atrial premature, junctional, nodal, and aberrant beats.",
    clinicalNote: "Originates above the ventricles. Often indicates atrial dysfunction or accessory pathways.",
    prevalence: "~8%",
    icon: "🟡",
  },
  {
    id: 2,
    name: "PVC",
    short: "VEB",
    color: "#ef4444",
    bgColor: "rgba(239,68,68,0.1)",
    borderColor: "rgba(239,68,68,0.3)",
    annotations: ["V", "E"],
    description:
      "Premature Ventricular Contractions and ventricular escape beats. Wide QRS complex morphology.",
    clinicalNote: "Ectopic ventricular origin. Associated with structural heart disease, electrolyte imbalance, and ischemia.",
    prevalence: "~6%",
    icon: "🔴",
  },
  {
    id: 3,
    name: "Global",
    short: "PCB",
    color: "#3b82f6",
    bgColor: "rgba(59,130,246,0.1)",
    borderColor: "rgba(59,130,246,0.3)",
    annotations: ["/", "f", "F", "Q", "!"],
    description:
      "Pacemaker-driven beats, fusion beats (normal + paced), and unclassifiable rhythms.",
    clinicalNote: "Includes artificial pacemaker stimuli and beats that don't fit standard morphological categories.",
    prevalence: "~16%",
    icon: "🔵",
  },
];

export const PIPELINE_STEPS = [
  {
    step: 1,
    icon: "📥",
    title: "Data Ingestion",
    subtitle: "MIT-BIH / CSV / EDF",
    description:
      "Supports the full 48-record MIT-BIH Arrhythmia Database from PhysioNet, plus custom CSV signals and EDF (European Data Format) files from clinical devices.",
    detail: "WFDB library handles MIT-BIH record parsing. ECGLoader auto-detects signal columns in CSV files.",
    color: "#6366f1",
  },
  {
    step: 2,
    icon: "🔍",
    title: "Clinical Audit",
    subtitle: "Rule-based QA",
    description:
      "SignalAuditor checks for missing values, consecutive duplicate samples (sensor freeze), flat-line segments, amplitude outliers, and sampling rate mismatches.",
    detail: "Reports: PASS / WARNING / CRITICAL severity. Thresholds: >20% missing → CRITICAL, any issues → WARNING.",
    color: "#8b5cf6",
  },
  {
    step: 3,
    icon: "🎛️",
    title: "Signal Denoising",
    subtitle: "Butterworth + Notch",
    description:
      "Butterworth bandpass filter (0.5–40 Hz) removes baseline wander and high-frequency muscle noise. IIR Notch filter at 60 Hz eliminates power-line interference.",
    detail: "4th-order bandpass, Quality factor 30 for notch. Zero-phase filtering via scipy.signal.filtfilt.",
    color: "#a855f7",
  },
  {
    step: 4,
    icon: "✂️",
    title: "Beat Segmentation",
    subtitle: "R-peak centering",
    description:
      "Each beat is extracted as a 360-sample (1-second) window centered on the R-peak annotation. Z-score normalized per-beat for amplitude invariance.",
    detail: "180 samples before + 180 samples after R-peak. For uploaded signals: scipy find_peaks auto-detection.",
    color: "#c084fc",
  },
  {
    step: 5,
    icon: "🧠",
    title: "CNN-LSTM Classification",
    subtitle: "Hybrid deep model",
    description:
      "1D CNN extracts morphological features from the waveform shape. BiLSTM models temporal dependencies in the feature sequence.",
    detail: "CNN: [1→64→128→256] channels, kernel sizes [5,5,3]. BiLSTM: 128 hidden × 2 directions = 256 features.",
    color: "#6366f1",
  },
  {
    step: 6,
    icon: "🔥",
    title: "Grad-CAM Explanation",
    subtitle: "Visual explainability",
    description:
      "Gradient-weighted Class Activation Mapping identifies which temporal regions of the ECG waveform (P-wave, QRS, T-wave, ST-segment) drove the prediction.",
    detail: "Hooks into the last Conv1D layer. Gradient global-average-pooled → channel weights → weighted activation sum.",
    color: "#f59e0b",
  },
  {
    step: 7,
    icon: "📄",
    title: "Clinical Report",
    subtitle: "HTML audit report",
    description:
      "Self-contained HTML report with the Grad-CAM overlay plot, confidence scores, signal audit summary, and model performance metrics — downloadable for review.",
    detail: "Matplotlib plot embedded as base64 PNG. Jinja2-rendered HTML. Saved to reports/ directory.",
    color: "#10b981",
  },
];

export const MODEL_ARCHITECTURES = [
  {
    id: "hybrid",
    name: "Hybrid CNN-LSTM",
    badge: "Default",
    badgeColor: "#6366f1",
    description: "The core architecture. 1D CNN extracts spatial waveform features, BiLSTM captures temporal rhythmic patterns.",
    pros: ["Best overall accuracy", "Fast convergence", "Balanced complexity"],
    layers: [
      { name: "Input", shape: "(B, 1, 360)", color: "#475569" },
      { name: "Conv1D(64) + BN + ReLU + Pool", shape: "(B, 64, 180)", color: "#6366f1" },
      { name: "Conv1D(128) + BN + ReLU + Pool", shape: "(B, 128, 90)", color: "#7c3aed" },
      { name: "Conv1D(256) + BN + ReLU + Pool ← Grad-CAM", shape: "(B, 256, 45)", color: "#9333ea" },
      { name: "BiLSTM(128 × 2 directions)", shape: "(B, 256)", color: "#a855f7" },
      { name: "FC(128) + BN + ReLU + Dropout(0.5)", shape: "(B, 128)", color: "#c084fc" },
      { name: "Softmax Output", shape: "(B, 4)", color: "#10b981" },
    ],
  },
  {
    id: "attention",
    name: "Attention Hybrid",
    badge: "Experimental",
    badgeColor: "#f59e0b",
    description: "Adds a self-attention mechanism between CNN and LSTM to focus on clinically relevant segments.",
    pros: ["Explicit attention weights", "Better on irregular rhythms", "Interpretable focus"],
    layers: [
      { name: "Input", shape: "(B, 1, 360)", color: "#475569" },
      { name: "CNN1DEncoder (same as Hybrid)", shape: "(B, 256, 45)", color: "#6366f1" },
      { name: "Self-Attention (Q·Kᵀ/√d · V)", shape: "(B, 45, 256)", color: "#f59e0b" },
      { name: "BiLSTM", shape: "(B, 256)", color: "#a855f7" },
      { name: "FC Head + Softmax", shape: "(B, 4)", color: "#10b981" },
    ],
  },
  {
    id: "resnet",
    name: "ResNet Hybrid",
    badge: "Deep",
    badgeColor: "#ef4444",
    description: "Residual connections prevent vanishing gradients in deeper networks. Skip connections add input to output of each block.",
    pros: ["Deeper feature extraction", "Residual skip connections", "Handles complex morphologies"],
    layers: [
      { name: "Input", shape: "(B, 1, 360)", color: "#475569" },
      { name: "Initial Conv(64, k=7) + Pool", shape: "(B, 64, 90)", color: "#6366f1" },
      { name: "ResBlock(64→64)", shape: "(B, 64, 90)", color: "#ef4444" },
      { name: "ResBlock(64→128, stride=2)", shape: "(B, 128, 45)", color: "#dc2626" },
      { name: "ResBlock(128→256, stride=2)", shape: "(B, 256, 23)", color: "#b91c1c" },
      { name: "BiLSTM + FC Head", shape: "(B, 4)", color: "#10b981" },
    ],
  },
  {
    id: "se",
    name: "SE-CNN Hybrid",
    badge: "Recalibrated",
    badgeColor: "#3b82f6",
    description: "Squeeze-and-Excitation blocks add channel-wise attention — the network learns which feature channels matter most.",
    pros: ["Channel-wise recalibration", "Learns feature importance", "Minimal parameter overhead"],
    layers: [
      { name: "Input", shape: "(B, 1, 360)", color: "#475569" },
      { name: "SECNNBlock: Conv + SE(AvgPool→FC→Sigmoid)", shape: "(B, 64, 180)", color: "#3b82f6" },
      { name: "SECNNBlock", shape: "(B, 128, 90)", color: "#2563eb" },
      { name: "SECNNBlock", shape: "(B, 256, 45)", color: "#1d4ed8" },
      { name: "BiLSTM + FC Head", shape: "(B, 4)", color: "#10b981" },
    ],
  },
];

export const TECH_STACK = [
  { name: "PyTorch", role: "Deep Learning Framework", icon: "🔥", color: "#ee4c2c" },
  { name: "Python", role: "Core Language", icon: "🐍", color: "#3776ab" },
  { name: "Streamlit", role: "Interactive Dashboard", icon: "⚡", color: "#ff4b4b" },
  { name: "WFDB", role: "Physiological Signal I/O", icon: "📡", color: "#6366f1" },
  { name: "SciPy", role: "Signal Processing Filters", icon: "📊", color: "#8ecae6" },
  { name: "NumPy", role: "Numerical Computing", icon: "🔢", color: "#013243" },
  { name: "scikit-learn", role: "Metrics & Evaluation", icon: "📐", color: "#f89939" },
  { name: "Plotly", role: "Interactive Charts", icon: "📈", color: "#3d4f7c" },
  { name: "MIT-BIH", role: "Reference ECG Dataset", icon: "🏥", color: "#10b981" },
  { name: "Grad-CAM", role: "XAI Explainability", icon: "🌡️", color: "#f59e0b" },
  { name: "Jinja2", role: "Clinical Report Templates", icon: "📋", color: "#b41717" },
  { name: "Matplotlib", role: "Report Visualizations", icon: "🎨", color: "#11557c" },
];

// Fake ECG waveform data for the hero animation (one beat of NSR)
export const ECG_WAVEFORM_DATA = Array.from({ length: 360 }, (_, i) => {
  const t = i / 360;
  // Construct a realistic-looking ECG beat
  const p = 0.18 * Math.exp(-((t - 0.12) ** 2) / (2 * 0.012 ** 2));        // P wave
  const q = -0.08 * Math.exp(-((t - 0.21) ** 2) / (2 * 0.006 ** 2));       // Q
  const r = 1.3 * Math.exp(-((t - 0.25) ** 2) / (2 * 0.008 ** 2));         // R (dominant)
  const s = -0.15 * Math.exp(-((t - 0.29) ** 2) / (2 * 0.007 ** 2));       // S
  const t_wave = 0.22 * Math.exp(-((t - 0.42) ** 2) / (2 * 0.022 ** 2));   // T wave
  const noise = (Math.random() - 0.5) * 0.015;
  return { x: i, y: parseFloat((p + q + r + s + t_wave + noise).toFixed(4)) };
});

// Grad-CAM style heatmap overlay for demo
export const GRADCAM_HEATMAP = Array.from({ length: 360 }, (_, i) => {
  const t = i / 360;
  // High attention near QRS complex (around sample 90), moderate at T-wave
  const qrs = Math.exp(-((t - 0.25) ** 2) / (2 * 0.025 ** 2));
  const twave = 0.4 * Math.exp(-((t - 0.42) ** 2) / (2 * 0.03 ** 2));
  return Math.min(1, qrs + twave);
});

export const STATS = [
  { label: "ECG Records", value: "48", suffix: "", description: "MIT-BIH database records" },
  { label: "Beat Segments", value: "100K+", suffix: "", description: "Processed heartbeat windows" },
  { label: "Arrhythmia Classes", value: "4", suffix: "", description: "Clinically meaningful categories" },
  { label: "Beat Window", value: "1", suffix: "sec", description: "360 samples @ 360 Hz" },
  { label: "CNN Layers", value: "3", suffix: "", description: "Progressively deeper feature maps" },
  { label: "BiLSTM Hidden", value: "128×2", suffix: "", description: "Bidirectional temporal encoding" },
];
