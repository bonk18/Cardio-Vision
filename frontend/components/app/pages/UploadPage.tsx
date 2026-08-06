"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Upload, FileUp, CheckCircle2, AlertTriangle, XCircle, Download, Loader2 } from "lucide-react";
import { uploadECG, downloadMITBIH, getPreprocessStatus, type UploadResponse, type StatusResponse } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

interface Props { status: StatusResponse | null }

export default function UploadPage({ status }: Props) {
  const [tab, setTab] = useState<"upload" | "mitbih">("upload");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [mitbihStatus, setMitbihStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [mitbihMsg, setMitbihMsg] = useState("");
  const [mitbihResults, setMitbihResults] = useState<{ total_beats: number; class_counts: Record<string, number> } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up poll on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const res = await uploadECG(file);
      setUploadResult(res);
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const startMITBIH = async () => {
    if (mitbihStatus === "running") return;
    setMitbihStatus("running");
    setMitbihMsg("Starting download...");
    setMitbihResults(null);
    try {
      const res = await downloadMITBIH() as { started: boolean; message?: string };
      if (!res.started) {
        setMitbihMsg(res.message ?? "Already running");
        return;
      }
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        const s = await getPreprocessStatus();
        setMitbihMsg(s.message);
        setMitbihStatus(s.status);
        if (s.results) setMitbihResults(s.results);
        if (s.status === "done" || s.status === "error") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
        }
      }, 2000);
    } catch (e: unknown) {
      setMitbihStatus("error");
      setMitbihMsg(e instanceof Error ? e.message : "Failed");
    }
  };

  const signal = uploadResult?.signal ?? [];
  const chartData = signal.map((y, x) => ({ x, y }));
  const audit = uploadResult?.audit;

  const severityColor = audit?.severity === "PASS" ? "#10b981"
    : audit?.severity === "WARNING" ? "#f59e0b" : "#ef4444";

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Upload &amp; Preprocess</h1>
        <p className="text-slate-500 mt-1 text-sm">Load ECG data, run clinical auditing, and segment beats</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-8 bg-[#0d1220] rounded-xl p-1 w-fit border border-indigo-500/10">
        {[{ id: "upload", label: "Upload File" }, { id: "mitbih", label: "MIT-BIH Database" }].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id as "upload" | "mitbih")}
            className={`px-5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              tab === t.id
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "upload" && (
        <div className="space-y-6">
          {/* Drop zone */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`glass-card p-12 text-center cursor-pointer transition-all duration-300 ${
              dragging ? "border-indigo-500/60 bg-indigo-500/10" : "border-white/10 hover:border-indigo-500/30"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.edf"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {uploading ? (
              <Loader2 className="mx-auto animate-spin text-indigo-400 mb-4" size={40} />
            ) : (
              <FileUp className="mx-auto text-slate-500 mb-4" size={40} />
            )}
            <p className="text-slate-300 font-medium mb-1">
              {uploading ? "Processing..." : "Drop CSV or EDF file here"}
            </p>
            <p className="text-xs text-slate-600">
              Supports MIT-BIH style CSV (single numeric column) and European Data Format (.edf)
            </p>
          </motion.div>

          {uploadError && (
            <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">
              <XCircle size={16} />
              {uploadError}
            </div>
          )}

          {uploadResult && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-5"
            >
              {/* Signal plot */}
              <div className="glass-card p-5 border border-indigo-500/15">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-sm font-semibold text-white">{uploadResult.filename}</p>
                  <span className="text-xs text-slate-500">{uploadResult.fs} Hz · {chartData.length} samples</span>
                </div>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                      <XAxis dataKey="x" hide />
                      <YAxis hide />
                      <Tooltip
                        contentStyle={{ background: "#0d1220", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8, fontSize: 11 }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(v: any) => [typeof v === "number" ? v.toFixed(4) : v, "Amplitude"]}
                        labelFormatter={() => ""}
                      />
                      <Line type="monotone" dataKey="y" stroke="#6366f1" strokeWidth={1.5} dot={false} animationDuration={1500} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Audit results */}
              {audit && (
                <div className="glass-card p-5 border" style={{ borderColor: severityColor + "30" }}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-white">Signal Quality Audit</h3>
                    <span
                      className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold"
                      style={{ background: severityColor + "15", color: severityColor }}
                    >
                      {audit.severity === "PASS" ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                      {audit.severity}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                    {[
                      { label: "Samples", value: audit.total_samples.toLocaleString() },
                      { label: "Duration", value: `${audit.duration_sec.toFixed(1)}s` },
                      { label: "Missing", value: `${audit.missing_pct.toFixed(1)}%` },
                      { label: "Flat Segments", value: audit.flat_segments },
                    ].map((m) => (
                      <div key={m.label} className="bg-[#060912] rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-white">{m.value}</div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">{m.label}</div>
                      </div>
                    ))}
                  </div>
                  {audit.issues.length > 0 ? (
                    <div className="space-y-1.5">
                      {audit.issues.map((iss, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-yellow-400 bg-yellow-500/5 rounded-lg px-3 py-2">
                          <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                          {iss}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-xs text-emerald-400">
                      <CheckCircle2 size={12} />
                      No issues detected — signal quality is good
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </div>
      )}

      {tab === "mitbih" && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="glass-card p-6 border border-indigo-500/15">
            <h3 className="text-base font-semibold text-white mb-2">MIT-BIH Arrhythmia Database</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-4">
              Downloads all 48 records from PhysioNet via WFDB, audits each recording,
              applies Butterworth bandpass + Notch denoising, segments R-peak centered
              beats (360 samples each), and saves a compressed .npz archive.
            </p>

            {status?.data_ready && (
              <div className="mb-4 flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 rounded-lg px-3 py-2">
                <CheckCircle2 size={13} />
                Data already processed — {(status.data_stats as { total_beats?: number })?.total_beats?.toLocaleString()} beats ready
              </div>
            )}

            <button
              onClick={startMITBIH}
              disabled={mitbihStatus === "running"}
              className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300"
            >
              {mitbihStatus === "running" ? (
                <><Loader2 size={15} className="animate-spin" /> Processing...</>
              ) : (
                <><Download size={15} /> Download &amp; Process MIT-BIH</>
              )}
            </button>
          </div>

          {mitbihStatus !== "idle" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={`glass-card p-5 border ${
                mitbihStatus === "done" ? "border-emerald-500/25"
                : mitbihStatus === "error" ? "border-red-500/25"
                : "border-indigo-500/20"
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                {mitbihStatus === "running" && <Loader2 size={15} className="animate-spin text-indigo-400" />}
                {mitbihStatus === "done" && <CheckCircle2 size={15} className="text-emerald-400" />}
                {mitbihStatus === "error" && <XCircle size={15} className="text-red-400" />}
                <span className="text-sm text-slate-300">{mitbihMsg}</span>
              </div>
              {mitbihResults && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                  {Object.entries(mitbihResults.class_counts).map(([name, count]) => (
                    <div key={name} className="bg-[#060912] rounded-xl p-3 text-center">
                      <div className="text-lg font-bold text-white">{count.toLocaleString()}</div>
                      <div className="text-[10px] text-slate-500">{name} beats</div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      )}
    </div>
  );
}
