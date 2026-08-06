const BASE = "http://localhost:8001";

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts?.headers ?? {}) },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "API error");
  }
  return res.json();
}

// ── Status ──────────────────────────────────────────────────
export const getStatus = () => req<StatusResponse>("/api/status");

// ── Upload ──────────────────────────────────────────────────
export async function uploadECG(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export const downloadMITBIH = () =>
  req("/api/download-mitbih", { method: "POST", headers: {} });

export const getPreprocessStatus = () =>
  req<PreprocessStatus>("/api/preprocess-status");

// ── Training ─────────────────────────────────────────────────
export const startTraining = (body: TrainRequest) =>
  req("/api/train", { method: "POST", body: JSON.stringify(body) });

export const getTrainStatus = () => req<TrainStatus>("/api/train/status");

// ── Inference ────────────────────────────────────────────────
export const getDatasetSample = (arch: string, classFilter: string, idx: number) =>
  req<SampleResponse>(
    `/api/dataset/sample?arch=${arch}&class_filter=${classFilter}&idx=${idx}`
  );

export async function runInference(
  arch: string,
  source: "dataset" | "upload",
  classFilter: string,
  idx: number,
  file?: File
): Promise<InferenceResponse> {
  const fd = new FormData();
  fd.append("arch", arch);
  fd.append("source", source);
  fd.append("class_filter", classFilter);
  fd.append("idx", String(idx));
  if (file) fd.append("file", file);

  const res = await fetch(`${BASE}/api/infer`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Inference failed");
  }
  return res.json();
}

// ── Reports ──────────────────────────────────────────────────
export const generateReport = (body: ReportRequest) =>
  req<{ path: string; filename: string }>("/api/report", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const listReports = () =>
  req<{ reports: { name: string; size: number }[] }>("/api/reports/list");

export const reportDownloadUrl = (filename: string) =>
  `${BASE}/api/report/download/${filename}`;

// ── Types ────────────────────────────────────────────────────
export interface StatusResponse {
  device: string;
  cuda: boolean;
  gpu_name: string | null;
  gpu_memory_gb: number | null;
  data_ready: boolean;
  data_stats: {
    total_beats: number;
    beat_window: number;
    class_counts: Record<string, number>;
  } | Record<string, never>;
  models: Record<string, { exists: boolean; epoch?: number | string; val_loss?: number; val_acc?: number }>;
  class_names: string[];
}

export interface UploadResponse {
  filename: string;
  signal: number[];
  fs: number;
  audit: AuditResult;
}

export interface AuditResult {
  record_id: string;
  total_samples: number;
  duration_sec: number;
  sampling_rate: number;
  missing_pct: number;
  duplicate_pct: number;
  flat_segments: number;
  amplitude_outliers: number;
  issues: string[];
  severity: "PASS" | "WARNING" | "CRITICAL";
}

export interface PreprocessStatus {
  status: "idle" | "running" | "done" | "error";
  message: string;
  results: { total_beats: number; class_counts: Record<string, number> } | null;
}

export interface TrainRequest {
  arch: string;
  epochs: number;
  batch_size: number;
  lr: number;
  seed: number;
}

export interface TrainStatus {
  running: boolean;
  epoch: number;
  total_epochs: number;
  train_loss: number[];
  val_loss: number[];
  train_acc: number[];
  val_acc: number[];
  status: "idle" | "running" | "done" | "error";
  message: string;
  results: {
    accuracy: number;
    macro_f1: number;
    confusion_matrix: number[][];
    classification_report: Record<string, Record<string, number>>;
  } | null;
}

export interface SampleResponse {
  signal: number[];
  true_label: string;
  record_id: string | null;
  total: number;
  idx: number;
}

export interface InferenceResponse {
  signal: number[];
  heatmap: number[];
  prediction: number;
  prediction_name: string;
  probabilities: number[];
  true_label: string | null;
  record_id: string | null;
}

export interface ReportRequest {
  signal: number[];
  heatmap: number[];
  prediction: number;
  probabilities: number[];
  record_id?: string;
  audit?: AuditResult | null;
  model_metrics?: object | null;
}
