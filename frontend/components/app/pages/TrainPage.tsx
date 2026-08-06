"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, Square, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { startTraining, getTrainStatus, type TrainStatus, type StatusResponse } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  CartesianGrid,
} from "recharts";

interface Props { selectedArch: string; status: StatusResponse | null; }

const CLASS_NAMES = ["Normal", "AFib", "PVC", "Global"];

export default function TrainPage({ selectedArch, status }: Props) {
  const [arch, setArch] = useState(selectedArch);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(128);
  const [lr, setLr] = useState(0.001);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stop polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const startPoll = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    // Immediately fetch once, then every 1.5s
    getTrainStatus().then(setTrainStatus).catch(() => {});
    pollRef.current = setInterval(async () => {
      try {
        const s = await getTrainStatus();
        setTrainStatus(s);
        if (!s.running && (s.status === "done" || s.status === "error")) {
          clearInterval(pollRef.current!);
          pollRef.current = null;
        }
      } catch { /* ignore transient errors */ }
    }, 1500);
  };

  const handleStart = async () => {
    setError(null);
    try {
      await startTraining({ arch, epochs, batch_size: batchSize, lr, seed: 42 });
      startPoll();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start training");
    }
  };

  const isRunning = trainStatus?.running ?? false;

  // Build chart data
  const chartData = (trainStatus?.train_loss ?? []).map((_, i) => ({
    epoch: i + 1,
    train_loss: trainStatus?.train_loss[i],
    val_loss: trainStatus?.val_loss[i],
    train_acc: trainStatus ? +((trainStatus.train_acc[i] ?? 0) * 100).toFixed(2) : 0,
    val_acc: trainStatus ? +((trainStatus.val_acc[i] ?? 0) * 100).toFixed(2) : 0,
  }));

  const results = trainStatus?.results;
  const cm = results?.confusion_matrix;

  const progress = trainStatus
    ? (trainStatus.epoch / Math.max(trainStatus.total_epochs, 1)) * 100
    : 0;

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Train Model</h1>
        <p className="text-slate-500 mt-1 text-sm">Configure hyperparameters and monitor training in real-time</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6 mb-6">
        {/* Config */}
        <div className="glass-card p-6 border border-indigo-500/10">
          <h3 className="text-sm font-semibold text-white mb-4">Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 block">Architecture</label>
              <select
                value={arch}
                onChange={(e) => setArch(e.target.value)}
                disabled={isRunning}
                className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none disabled:opacity-50"
              >
                {["hybrid", "attention", "resnet", "se"].map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </div>
            {[
              { label: "Epochs", value: epochs, set: setEpochs, min: 1, max: 200, step: 1 },
              { label: "Batch Size", value: batchSize, set: setBatchSize, min: 32, max: 512, step: 32 },
            ].map((field) => (
              <div key={field.label}>
                <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 block">{field.label}</label>
                <input
                  type="number"
                  value={field.value}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  disabled={isRunning}
                  onChange={(e) => field.set(Number(e.target.value))}
                  className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none disabled:opacity-50"
                />
              </div>
            ))}
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 block">Learning Rate</label>
              <input
                type="number"
                value={lr}
                min={1e-5}
                max={0.1}
                step={1e-4}
                disabled={isRunning}
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none disabled:opacity-50"
              />
            </div>
          </div>

          {error && (
            <div className="mt-4 flex items-start gap-2 text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
              <XCircle size={12} className="mt-0.5 flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            onClick={handleStart}
            disabled={isRunning}
            className="mt-5 w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white disabled:opacity-50 hover:shadow-lg hover:shadow-indigo-500/25 transition-all"
          >
            {isRunning ? (
              <><Loader2 size={15} className="animate-spin" /> Training...</>
            ) : (
              <><Play size={15} fill="white" /> Start Training</>
            )}
          </button>
        </div>

        {/* Status panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Progress */}
          {trainStatus && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`glass-card p-5 border ${
                trainStatus.status === "done" ? "border-emerald-500/25"
                : trainStatus.status === "error" ? "border-red-500/25"
                : "border-indigo-500/20"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  {isRunning && <Loader2 size={14} className="animate-spin text-indigo-400" />}
                  {trainStatus.status === "done" && <CheckCircle2 size={14} className="text-emerald-400" />}
                  {trainStatus.status === "error" && <XCircle size={14} className="text-red-400" />}
                  <span className="text-sm text-slate-300">{trainStatus.message}</span>
                </div>
                <span className="text-xs text-slate-500">
                  {trainStatus.epoch}/{trainStatus.total_epochs}
                </span>
              </div>
              <div className="h-2 bg-[#060912] rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </motion.div>
          )}

          {/* Live loss chart */}
          {chartData.length > 0 && (
            <div className="glass-card p-5 border border-indigo-500/10">
              <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-4">Loss Curves</h4>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 2, right: 5, left: -25, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="epoch" tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#0d1220", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8, fontSize: 11 }} />
                    <Legend iconSize={8} wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
                    <Line type="monotone" dataKey="train_loss" stroke="#6366f1" strokeWidth={1.5} dot={false} name="Train Loss" />
                    <Line type="monotone" dataKey="val_loss" stroke="#a78bfa" strokeWidth={1.5} dot={false} name="Val Loss" strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {chartData.length > 0 && (
            <div className="glass-card p-5 border border-indigo-500/10">
              <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-4">Accuracy</h4>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 2, right: 5, left: -25, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="epoch" tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                    <Tooltip contentStyle={{ background: "#0d1220", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8, fontSize: 11 }} />
                    <Legend iconSize={8} wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
                    <Line type="monotone" dataKey="train_acc" stroke="#10b981" strokeWidth={1.5} dot={false} name="Train Acc" />
                    <Line type="monotone" dataKey="val_acc" stroke="#34d399" strokeWidth={1.5} dot={false} name="Val Acc" strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results + Confusion matrix */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6 border border-emerald-500/20"
        >
          <div className="flex items-center gap-3 mb-6">
            <CheckCircle2 size={18} className="text-emerald-400" />
            <h3 className="text-base font-semibold text-white">
              Training Complete — Test Accuracy:{" "}
              <span className="text-emerald-400">{(results.accuracy * 100).toFixed(1)}%</span>
              {" "}· Macro F1:{" "}
              <span className="text-indigo-400">{(results.macro_f1 * 100).toFixed(1)}%</span>
            </h3>
          </div>

          {cm && (
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Confusion Matrix</p>
              <div className="overflow-x-auto">
                <table className="text-xs">
                  <thead>
                    <tr>
                      <th className="px-3 py-2 text-slate-600 font-normal text-right">Predicted →</th>
                      {CLASS_NAMES.map((n) => (
                        <th key={n} className="px-4 py-2 text-slate-400 font-semibold text-center">{n}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cm.map((row, i) => {
                      const rowSum = row.reduce((a, b) => a + b, 0);
                      return (
                        <tr key={i}>
                          <td className="px-3 py-2 text-slate-400 font-semibold text-right">{CLASS_NAMES[i]}</td>
                          {row.map((val, j) => {
                            const pct = rowSum > 0 ? val / rowSum : 0;
                            const bg = i === j
                              ? `rgba(16,185,129,${0.1 + pct * 0.5})`
                              : pct > 0.1 ? `rgba(239,68,68,${pct * 0.5})` : "transparent";
                            return (
                              <td key={j} className="px-4 py-2 text-center rounded" style={{ background: bg }}>
                                <span className={i === j ? "text-emerald-400 font-bold" : "text-slate-400"}>
                                  {val}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
