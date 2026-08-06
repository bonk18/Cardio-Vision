"use client";

import { motion } from "framer-motion";
import { ExternalLink, ChevronDown, Zap, Shield, Brain } from "lucide-react";
import { ECG_WAVEFORM_DATA, GRADCAM_HEATMAP } from "@/lib/data";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const pills = [
  { icon: <Brain size={12} />, label: "Hybrid CNN-LSTM" },
  { icon: <Zap size={12} />, label: "Grad-CAM XAI" },
  { icon: <Shield size={12} />, label: "MIT-BIH Validated" },
];

// Blend ECG data with a heatmap color for the hero chart
const chartData = ECG_WAVEFORM_DATA.map((d, i) => ({
  ...d,
  heat: GRADCAM_HEATMAP[i],
}));

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col justify-center overflow-hidden grid-bg noise-bg">
      {/* Background blobs */}
      <div className="absolute top-1/4 -left-40 w-[600px] h-[600px] bg-indigo-600/8 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 -right-40 w-[500px] h-[500px] bg-purple-600/8 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6 pt-32 pb-24 w-full">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left — copy */}
          <div>
            {/* Pills */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex flex-wrap gap-2 mb-8"
            >
              {pills.map((p) => (
                <span
                  key={p.label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-indigo-500/10 border border-indigo-500/20 text-indigo-300"
                >
                  {p.icon}
                  {p.label}
                </span>
              ))}
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-5xl md:text-6xl lg:text-7xl font-extrabold leading-[1.08] tracking-tight mb-6"
            >
              <span className="text-white">Cardio</span>
              <span className="gradient-text">Vision</span>
              <br />
              <span className="text-3xl md:text-4xl lg:text-5xl text-slate-300 font-semibold">
                AI Cardiac Analysis
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-slate-400 text-lg leading-relaxed mb-10 max-w-lg"
            >
              An end-to-end explainable deep learning platform for clinical ECG
              arrhythmia detection — from raw signal to Grad-CAM visual
              explanation in a single pipeline.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-wrap gap-4"
            >
              <a
                href="/app"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:shadow-xl hover:shadow-indigo-500/30 transition-all duration-300 hover:-translate-y-1 text-sm"
              >
                Launch Streamlit App
                <ExternalLink size={15} />
              </a>
              <a
                href="#pipeline"
                className="flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold border border-white/10 text-slate-300 hover:text-white hover:bg-white/5 hover:border-white/20 transition-all duration-300 text-sm"
              >
                Explore Pipeline
                <ChevronDown size={15} />
              </a>
            </motion.div>

            {/* Mini stats row */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
              className="mt-12 flex flex-wrap gap-8"
            >
              {[
                { v: "48", l: "ECG Records" },
                { v: "4", l: "Arrhythmia Classes" },
                { v: "100K+", l: "Beat Segments" },
                { v: "360Hz", l: "Sampling Rate" },
              ].map((s) => (
                <div key={s.l}>
                  <div className="text-2xl font-bold gradient-text">{s.v}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{s.l}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right — ECG chart card */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="relative"
          >
            {/* Main card */}
            <div className="glass-card glow-border p-6 relative overflow-hidden">
              {/* Card header */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">
                    MIT-BIH Record 100 · Lead MLII
                  </p>
                  <p className="text-sm font-semibold text-slate-200">
                    Normal Sinus Rhythm · Beat #42
                  </p>
                </div>
                <span className="px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
                  ✓ Normal
                </span>
              </div>

              {/* ECG chart with Grad-CAM heatmap overlay */}
              <div className="relative h-52">
                {/* Heatmap background regions */}
                <div
                  className="absolute inset-0 pointer-events-none z-0"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent 20%, rgba(239,68,68,0.06) 40%, rgba(239,68,68,0.12) 48%, rgba(239,68,68,0.08) 56%, transparent 65%)",
                    borderRadius: 8,
                  }}
                />
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                    <XAxis dataKey="x" hide />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1220",
                        border: "1px solid rgba(99,102,241,0.3)",
                        borderRadius: 8,
                        fontSize: 11,
                        color: "#e2e8f0",
                      }}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(v: any) => [
                      typeof v === "number" ? v.toFixed(3) : String(v ?? ""),
                      "Amplitude",
                    ]}
                      labelFormatter={() => ""}
                    />
                    <Line
                      type="monotone"
                      dataKey="y"
                      stroke="#6366f1"
                      strokeWidth={1.5}
                      dot={false}
                      animationDuration={2500}
                      animationEasing="ease-out"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Grad-CAM legend */}
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <div className="w-3 h-3 rounded-sm bg-red-500/40" />
                  High Grad-CAM attention (QRS)
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                  <span>360 samples · 1 sec</span>
                  <span className="text-indigo-400">360 Hz</span>
                </div>
              </div>
            </div>

            {/* Floating confidence card */}
            <motion.div
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute -bottom-6 -left-8 glass-card px-4 py-3 shadow-2xl"
            >
              <p className="text-xs text-slate-500 mb-1">Confidence</p>
              <div className="flex items-end gap-1">
                <span className="text-2xl font-bold text-emerald-400">98.4%</span>
                <span className="text-xs text-slate-500 mb-0.5">Normal</span>
              </div>
              <div className="mt-1.5 h-1.5 w-32 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                  style={{ width: "98.4%" }}
                />
              </div>
            </motion.div>

            {/* Floating device badge */}
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
              className="absolute -top-4 -right-4 glass-card px-3 py-2 text-xs"
            >
              <span className="text-emerald-400 font-semibold">● CPU</span>
              <span className="text-slate-500 ml-2">/ CUDA Ready</span>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.a
        href="#overview"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1, y: [0, 8, 0] }}
        transition={{ opacity: { delay: 1 }, y: { duration: 2, repeat: Infinity } }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-slate-600 hover:text-slate-400 transition-colors"
      >
        <span className="text-xs tracking-widest uppercase">Scroll</span>
        <ChevronDown size={16} />
      </motion.a>
    </section>
  );
}
