"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FileText, Download, Loader2, AlertTriangle, RefreshCw } from "lucide-react";
import { generateReport, listReports, type InferenceResponse } from "@/lib/api";

const BASE_API = "http://localhost:8001";

// Programmatically trigger a file download (respects Content-Disposition: attachment)
async function triggerDownload(filename: string) {
  const url = `${BASE_API}/api/report/download/${encodeURIComponent(filename)}`;
  const res = await fetch(url);
  if (!res.ok) return;
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

interface Props {
  lastInference: InferenceResponse | null;
}

interface ReportEntry { name: string; size: number; }

const CLASS_NAMES = ["Normal", "AFib", "PVC", "Global"];
const CLASS_COLORS: Record<string, string> = {
  Normal: "#10b981", AFib: "#f59e0b", PVC: "#ef4444", Global: "#3b82f6",
};

export default function ReportsPage({ lastInference }: Props) {
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [latestFilename, setLatestFilename] = useState<string | null>(null);
  const [reports, setReports] = useState<ReportEntry[]>([]);
  const [loadingList, setLoadingList] = useState(false);

  const fetchList = async () => {
    setLoadingList(true);
    try {
      const res = await listReports();
      setReports(res.reports);
    } catch { /* ignore */ } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => { fetchList(); }, []);

  const handleGenerate = async () => {
    if (!lastInference) return;
    setGenerating(true);
    setGenError(null);
    try {
      const res = await generateReport({
        signal: lastInference.signal,
        heatmap: lastInference.heatmap,
        prediction: lastInference.prediction,
        probabilities: lastInference.probabilities,
        record_id: lastInference.record_id ?? "interactive_session",
      });
      setLatestFilename(res.filename);
      await fetchList();
    } catch (e: unknown) {
      setGenError(e instanceof Error ? e.message : "Failed to generate report");
    } finally {
      setGenerating(false);
    }
  };

  const inf = lastInference;
  const predColor = inf ? (CLASS_COLORS[inf.prediction_name] ?? "#6366f1") : "#6366f1";

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Clinical Reports</h1>
        <p className="text-slate-500 mt-1 text-sm">Generate and download HTML clinical audit reports</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Generate panel */}
        <div className="space-y-5">
          <div className="glass-card p-6 border border-indigo-500/10">
            <h3 className="text-sm font-semibold text-white mb-4">Generate Report</h3>

            {!inf ? (
              <div className="flex items-start gap-3 text-sm text-yellow-400 bg-yellow-500/10 rounded-xl p-4 border border-yellow-500/20">
                <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
                Run inference on a sample first (Inference &amp; Explain page) before generating a report.
              </div>
            ) : (
              <>
                {/* Last inference summary */}
                <div className="glass-card p-4 border mb-5" style={{ borderColor: predColor + "30" }}>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Last Inference</p>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xl font-bold" style={{ color: predColor }}>
                        {inf.prediction_name}
                      </span>
                      {inf.record_id && (
                        <p className="text-xs text-slate-500 mt-0.5">Record: {inf.record_id}</p>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold" style={{ color: predColor }}>
                        {(inf.probabilities[inf.prediction] * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-slate-500">confidence</div>
                    </div>
                  </div>

                  {/* Class probs */}
                  <div className="mt-3 space-y-1.5">
                    {CLASS_NAMES.map((name, i) => (
                      <div key={name} className="flex items-center gap-2 text-xs">
                        <span className="w-12 text-slate-500 text-right">{name}</span>
                        <div className="flex-1 h-1.5 bg-[#060912] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${inf.probabilities[i] * 100}%`,
                              background: CLASS_COLORS[name],
                            }}
                          />
                        </div>
                        <span className="w-10 font-mono" style={{ color: CLASS_COLORS[name] }}>
                          {(inf.probabilities[i] * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {genError && (
                  <div className="mb-4 flex items-start gap-2 text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
                    <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" /> {genError}
                  </div>
                )}

                <button
                  onClick={handleGenerate}
                  disabled={generating}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white disabled:opacity-50 hover:shadow-lg hover:shadow-indigo-500/25 transition-all"
                >
                  {generating
                    ? <><Loader2 size={15} className="animate-spin" /> Generating...</>
                    : <><FileText size={15} /> Generate Clinical Report</>
                  }
                </button>

                {latestFilename && (
                  <button
                    onClick={() => triggerDownload(latestFilename)}
                    className="mt-3 w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10 transition-all"
                  >
                    <Download size={15} />
                    Download: {latestFilename}
                  </button>
                )}
              </>
            )}
          </div>

          {/* What's in the report */}
          <div className="glass-card p-5 border border-indigo-500/10">
            <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">Report Contents</h4>
            {[
              "Classification result with confidence score",
              "Grad-CAM ECG heatmap (matplotlib, embedded PNG)",
              "Class probability bars",
              "Signal quality audit summary",
              "Model performance metrics (if available)",
            ].map((item) => (
              <div key={item} className="flex items-center gap-2 text-xs text-slate-400 py-1.5 border-b border-white/5 last:border-0">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                {item}
              </div>
            ))}
          </div>
        </div>

        {/* Report list */}
        <div className="glass-card p-6 border border-indigo-500/10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Generated Reports</h3>
            <button
              onClick={fetchList}
              disabled={loadingList}
              className="p-1.5 rounded-lg hover:bg-white/5 text-slate-500 hover:text-slate-300 transition-colors"
            >
              <RefreshCw size={13} className={loadingList ? "animate-spin" : ""} />
            </button>
          </div>

          {reports.length === 0 ? (
            <div className="text-center py-12 text-slate-600 text-sm">
              No reports yet
            </div>
          ) : (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
              {reports.map((r, i) => (
                <motion.div
                  key={r.name}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-center justify-between bg-[#060912] rounded-xl px-4 py-3 border border-white/5 hover:border-white/10 transition-colors group"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <FileText size={14} className="text-indigo-400 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs text-slate-300 truncate">{r.name}</p>
                      <p className="text-[10px] text-slate-600">{(r.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </div>
                  <button
                    onClick={() => triggerDownload(r.name)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-indigo-500/20 text-indigo-400 transition-all"
                  >
                    <Download size={13} />
                  </button>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
