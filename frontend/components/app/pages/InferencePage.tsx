"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Loader2, FlaskConical, Upload, AlertTriangle } from "lucide-react";
import {
  runInference, getDatasetSample,
  type InferenceResponse, type StatusResponse,
} from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip,
  AreaChart, Area,
} from "recharts";

interface Props {
  selectedArch: string;
  status: StatusResponse | null;
  lastInference: InferenceResponse | null;
  setLastInference: (r: InferenceResponse) => void;
}

const CLASS_NAMES = ["Normal", "AFib", "PVC", "Global"];
const CLASS_COLORS: Record<string, string> = {
  Normal: "#10b981", AFib: "#f59e0b", PVC: "#ef4444", Global: "#3b82f6",
};

export default function InferencePage({ selectedArch, status, lastInference, setLastInference }: Props) {
  const [source, setSource] = useState<"dataset" | "upload">("dataset");
  const [classFilter, setClassFilter] = useState("Any");
  const [sampleIdx, setSampleIdx] = useState(0);
  const [sampleInfo, setSampleInfo] = useState<{ total: number; true_label: string; record_id: string | null } | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const modelExists = status?.models?.[selectedArch]?.exists ?? false;

  const loadSampleInfo = async (idx: number) => {
    try {
      const s = await getDatasetSample(selectedArch, classFilter, idx);
      setSampleInfo({ total: s.total, true_label: s.true_label, record_id: s.record_id });
    } catch { /* ignore */ }
  };

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runInference(selectedArch, source, classFilter, sampleIdx, uploadFile ?? undefined);
      setLastInference(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setRunning(false);
    }
  };

  const result = lastInference;
  const signalData = result?.signal.map((y, x) => ({ x, y })) ?? [];
  const heatData = result?.heatmap.map((h, x) => ({ x, h })) ?? [];
  const overlayData = result?.signal.map((y, x) => ({
    x, y,
    yHigh: (result.heatmap[x] ?? 0) > 0.4 ? y : null,
  })) ?? [];

  const predColor = result ? (CLASS_COLORS[result.prediction_name] ?? "#6366f1") : "#6366f1";

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Inference &amp; Explain</h1>
        <p className="text-slate-500 mt-1 text-sm">Run inference and visualize Grad-CAM heatmap</p>
      </div>

      {!modelExists && (
        <div className="mb-6 flex items-center gap-3 bg-yellow-500/10 border border-yellow-500/25 rounded-xl px-4 py-3 text-sm text-yellow-400">
          <AlertTriangle size={15} />
          No trained model found for <strong>{selectedArch}</strong>. Go to Train Model first.
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6 mb-6">
        {/* Controls */}
        <div className="glass-card p-6 border border-indigo-500/10 space-y-5">
          <h3 className="text-sm font-semibold text-white">Input Signal</h3>

          {/* Source tabs */}
          <div className="flex gap-1 bg-[#060912] rounded-xl p-1">
            {(["dataset", "upload"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setSource(s)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all ${
                  source === s ? "bg-indigo-500/20 text-indigo-300" : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {s === "upload" ? <Upload size={12} /> : <FlaskConical size={12} />}
                {s === "dataset" ? "Dataset" : "Upload"}
              </button>
            ))}
          </div>

          {source === "dataset" ? (
            <>
              <div>
                <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 block">Filter by Class</label>
                <select
                  value={classFilter}
                  onChange={(e) => { setClassFilter(e.target.value); loadSampleInfo(sampleIdx); }}
                  className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none"
                >
                  {["Any", ...CLASS_NAMES].map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 flex justify-between">
                  <span>Sample Index</span>
                  {sampleInfo && <span className="text-slate-600">of {sampleInfo.total.toLocaleString()}</span>}
                </label>
                <input
                  type="number"
                  value={sampleIdx}
                  min={0}
                  onChange={(e) => { const v = Number(e.target.value); setSampleIdx(v); loadSampleInfo(v); }}
                  className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none"
                />
              </div>
              {sampleInfo && (
                <div className="text-xs text-slate-500">
                  True label: <span className="font-semibold" style={{ color: CLASS_COLORS[sampleInfo.true_label] }}>{sampleInfo.true_label}</span>
                  {sampleInfo.record_id && <> · Record: <span className="text-slate-400">{sampleInfo.record_id}</span></>}
                </div>
              )}
            </>
          ) : (
            <>
              <div
                className="border-2 border-dashed border-indigo-500/20 rounded-xl p-6 text-center cursor-pointer hover:border-indigo-500/40 transition-colors"
                onClick={() => fileRef.current?.click()}
              >
                <input ref={fileRef} type="file" accept=".csv,.edf" className="hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
                <Upload size={24} className="mx-auto text-slate-600 mb-2" />
                <p className="text-xs text-slate-500">
                  {uploadFile ? uploadFile.name : "Click to select CSV or EDF"}
                </p>
              </div>
              {uploadFile && (
                <div>
                  <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 block">Beat Index</label>
                  <input type="number" value={sampleIdx} min={0}
                    onChange={(e) => setSampleIdx(Number(e.target.value))}
                    className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none" />
                </div>
              )}
            </>
          )}

          {error && (
            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            onClick={handleRun}
            disabled={running || !modelExists || (source === "upload" && !uploadFile)}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white disabled:opacity-50 hover:shadow-lg hover:shadow-indigo-500/25 transition-all"
          >
            {running ? <><Loader2 size={15} className="animate-spin" /> Computing...</>
              : <><FlaskConical size={15} /> Run Inference + Grad-CAM</>}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
              {/* Prediction header */}
              <div className="glass-card p-5 border" style={{ borderColor: predColor + "40" }}>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Prediction</p>
                    <div className="flex items-center gap-3">
                      <span className="text-3xl font-bold" style={{ color: predColor }}>
                        {result.prediction_name}
                      </span>
                      {result.true_label && (
                        <span className="text-xs text-slate-500">
                          (True: <span style={{ color: CLASS_COLORS[result.true_label] }}>{result.true_label}</span>)
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-500 mb-1">Confidence</p>
                    <p className="text-2xl font-bold" style={{ color: predColor }}>
                      {(result.probabilities[result.prediction] * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Probability bars */}
                <div className="mt-4 space-y-2">
                  {CLASS_NAMES.map((name, i) => {
                    const pct = result.probabilities[i] * 100;
                    const color = CLASS_COLORS[name];
                    return (
                      <div key={name} className="flex items-center gap-3 text-xs">
                        <span className="w-14 text-slate-400 text-right">{name}</span>
                        <div className="flex-1 h-2.5 bg-[#060912] rounded-full overflow-hidden">
                          <motion.div
                            className="h-full rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{ duration: 0.6, delay: i * 0.07 }}
                            style={{ background: color }}
                          />
                        </div>
                        <span className="w-12 font-mono font-semibold" style={{ color }}>
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* ECG + Grad-CAM overlay */}
              <div className="glass-card p-5 border border-red-500/15">
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">ECG + Grad-CAM overlay (red = high attention)</p>
                <div className="h-40 relative">
                  <div className="absolute inset-0 pointer-events-none rounded-lg"
                    style={{
                      background: "linear-gradient(90deg,transparent 20%,rgba(239,68,68,0.06) 38%,rgba(239,68,68,0.16) 46%,rgba(239,68,68,0.07) 58%,transparent 68%)"
                    }} />
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={signalData} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                      <XAxis dataKey="x" hide />
                      <YAxis hide />
                      <Tooltip contentStyle={{ background: "#0d1220", border: "1px solid rgba(99,102,241,0.25)", borderRadius: 8, fontSize: 11 }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(v: any) => [typeof v === "number" ? v.toFixed(4) : v, "Amplitude"]}
                        labelFormatter={() => ""} />
                      <Line type="monotone" dataKey="y" stroke="#ef4444" strokeWidth={1.5} dot={false} animationDuration={800} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Heatmap */}
              <div className="glass-card p-5 border border-amber-500/15">
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Grad-CAM attention heatmap</p>
                <div className="h-28">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={heatData} margin={{ top: 2, right: 5, left: -30, bottom: 2 }}>
                      <defs>
                        <linearGradient id="heatG" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="x" hide />
                      <YAxis hide domain={[0, 1]} />
                      <Area type="monotone" dataKey="h" stroke="#ef4444" strokeWidth={1.5} fill="url(#heatG)" dot={false} animationDuration={800} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex justify-between text-[10px] text-slate-600 mt-1">
                  <span>Low attention</span>
                  <span>High attention (QRS focus)</span>
                </div>
              </div>

              {/* Raw ECG */}
              <div className="glass-card p-5 border border-indigo-500/10">
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Raw ECG signal</p>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={signalData} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                      <XAxis dataKey="x" hide />
                      <YAxis hide />
                      <Line type="monotone" dataKey="y" stroke="#6366f1" strokeWidth={1.5} dot={false} animationDuration={800} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="glass-card h-64 flex items-center justify-center text-slate-600 text-sm border border-white/5">
              Run inference to see results here
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
