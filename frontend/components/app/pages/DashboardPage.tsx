"use client";

import { motion } from "framer-motion";
import { CheckCircle2, XCircle, Database, Brain, Cpu, Zap, ArrowRight } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface Props {
  status: StatusResponse | null;
  apiError: boolean;
  selectedArch: string;
  classColors: Record<string, string>;
  onNavigate: (page: string) => void;
}

const ARCHS = ["hybrid", "attention", "resnet", "se"];

export default function DashboardPage({ status, apiError, selectedArch, classColors, onNavigate }: Props) {
  const modelExists = status?.models?.[selectedArch]?.exists ?? false;
  const dataReady = status?.data_ready ?? false;

  const classCountData = status?.data_stats && "class_counts" in status.data_stats
    ? Object.entries(status.data_stats.class_counts).map(([name, count]) => ({ name, count }))
    : [];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-500 mt-1 text-sm">CardioVision — Hybrid CNN-LSTM Arrhythmia Detection</p>
      </div>

      {/* API error banner */}
      {apiError && (
        <div className="mb-6 flex items-center gap-3 bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">
          <XCircle size={16} />
          Cannot reach API at localhost:8000. Start the backend with:
          <code className="font-mono bg-red-500/10 px-2 py-0.5 rounded text-xs ml-1">
            python -m uvicorn api:app --reload
          </code>
        </div>
      )}

      {/* Status cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          {
            label: "Arrhythmia Classes",
            value: "4",
            sub: "Normal · AFib · PVC · Global",
            icon: <Brain size={18} />,
            color: "#6366f1",
          },
          {
            label: "Dataset",
            value: dataReady
              ? `${(status?.data_stats as { total_beats?: number })?.total_beats?.toLocaleString() ?? "—"}`
              : "Not ready",
            sub: dataReady ? "beats segmented" : "Run Download & Preprocess",
            icon: <Database size={18} />,
            color: dataReady ? "#10b981" : "#94a3b8",
            action: !dataReady ? () => onNavigate("upload") : undefined,
          },
          {
            label: `Model (${selectedArch})`,
            value: modelExists ? "Trained" : "Not trained",
            sub: modelExists
              ? `val_acc ${((status?.models?.[selectedArch]?.val_acc ?? 0) * 100).toFixed(1)}%`
              : "Go to Train page",
            icon: <Brain size={18} />,
            color: modelExists ? "#10b981" : "#94a3b8",
            action: !modelExists ? () => onNavigate("train") : undefined,
          },
          {
            label: "Compute",
            value: status ? (status.cuda ? "CUDA" : "CPU") : "—",
            sub: status?.gpu_name ?? (status?.cuda ? "" : "No GPU detected"),
            icon: status?.cuda ? <Zap size={18} /> : <Cpu size={18} />,
            color: status?.cuda ? "#10b981" : "#f59e0b",
          },
        ].map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07 }}
            onClick={card.action}
            className={`glass-card p-5 ${card.action ? "cursor-pointer hover:-translate-y-0.5 hover:shadow-lg transition-all duration-200" : ""}`}
            style={{ border: `1px solid ${card.color}25` }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-slate-500 uppercase tracking-wider">{card.label}</span>
              <div className="p-1.5 rounded-lg" style={{ background: card.color + "15", color: card.color }}>
                {card.icon}
              </div>
            </div>
            <div className="text-2xl font-bold" style={{ color: card.color }}>{card.value}</div>
            <div className="text-xs text-slate-500 mt-1 flex items-center gap-1">
              {card.sub}
              {card.action && <ArrowRight size={11} />}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* Class distribution chart */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6 border border-indigo-500/10"
        >
          <h3 className="text-sm font-semibold text-white mb-4">Beat Class Distribution</h3>
          {classCountData.length > 0 ? (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classCountData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                  <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#0d1220", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8, fontSize: 12 }}
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {classCountData.map((entry) => (
                      <Cell key={entry.name} fill={classColors[entry.name] ?? "#6366f1"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-slate-600 text-sm">
              No data — download & preprocess MIT-BIH first
            </div>
          )}
        </motion.div>

        {/* Architecture panel */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="glass-card p-6 border border-indigo-500/10"
        >
          <h3 className="text-sm font-semibold text-white mb-4">CNN-LSTM Architecture</h3>
          <div className="space-y-2 text-xs font-mono">
            {[
              { label: "Input", shape: "(B, 1, 360)", color: "#475569" },
              { label: "Conv1D(64) → BN → ReLU → Pool", shape: "(B, 64, 180)", color: "#6366f1" },
              { label: "Conv1D(128) → BN → ReLU → Pool", shape: "(B, 128, 90)", color: "#7c3aed" },
              { label: "Conv1D(256) → BN → ReLU → Pool  ← Grad-CAM", shape: "(B, 256, 45)", color: "#9333ea" },
              { label: "BiLSTM(128 × 2)", shape: "(B, 256)", color: "#a855f7" },
              { label: "FC(128) → Dropout(0.5)", shape: "(B, 128)", color: "#c084fc" },
              { label: "Softmax", shape: "(B, 4)", color: "#10b981" },
            ].map((row, i) => (
              <div key={i} className="flex items-center justify-between gap-3 px-3 py-1.5 rounded-lg" style={{ background: row.color + "10" }}>
                <span className="text-slate-300">{row.label}</span>
                <span className="px-2 py-0.5 rounded text-[10px]" style={{ background: row.color + "20", color: row.color }}>{row.shape}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Model status grid */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-6 border border-indigo-500/10"
      >
        <h3 className="text-sm font-semibold text-white mb-4">All Model Checkpoints</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {ARCHS.map((arch) => {
            const m = status?.models?.[arch];
            return (
              <div key={arch} className="bg-[#060912] rounded-xl p-4 border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-300 capitalize">{arch}</span>
                  {m?.exists ? (
                    <CheckCircle2 size={14} className="text-emerald-400" />
                  ) : (
                    <XCircle size={14} className="text-slate-600" />
                  )}
                </div>
                {m?.exists ? (
                  <>
                    <div className="text-lg font-bold text-emerald-400">
                      {((m.val_acc ?? 0) * 100).toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-slate-500">val acc · epoch {m.epoch}</div>
                  </>
                ) : (
                  <div className="text-xs text-slate-600 mt-1">Not trained</div>
                )}
              </div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
}
