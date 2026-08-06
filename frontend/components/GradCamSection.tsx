"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { ECG_WAVEFORM_DATA, GRADCAM_HEATMAP } from "@/lib/data";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from "recharts";

// Merge signal + heatmap
const signalWithHeat = ECG_WAVEFORM_DATA.map((d, i) => ({
  ...d,
  heat: GRADCAM_HEATMAP[i],
  // Attention-weighted signal color hint
  attention: GRADCAM_HEATMAP[i] > 0.4 ? d.y : null,
}));

export default function GradCamSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="gradcam" className="py-28 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left — explanation */}
          <motion.div
            ref={ref}
            initial={{ opacity: 0, x: -40 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7 }}
          >
            <span className="text-xs font-semibold text-amber-400 uppercase tracking-widest mb-4 block">
              Explainability
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 tracking-tight leading-tight">
              Grad-CAM makes the{" "}
              <span className="text-amber-400">model transparent</span>
            </h2>
            <p className="text-slate-400 text-base leading-relaxed mb-8">
              Standard neural networks are black boxes. CardioVision uses{" "}
              <strong className="text-slate-200">
                Gradient-weighted Class Activation Mapping (Grad-CAM)
              </strong>{" "}
              to explain{" "}
              <em>why</em> a beat was classified a certain way.
            </p>

            {/* Steps */}
            <div className="space-y-5">
              {[
                {
                  step: "01",
                  title: "Forward pass",
                  desc: "Feed the ECG beat through the network. The last Conv1D layer produces feature maps.",
                  color: "#6366f1",
                },
                {
                  step: "02",
                  title: "Backward gradients",
                  desc: "Compute gradients of the target class score with respect to those feature maps.",
                  color: "#a855f7",
                },
                {
                  step: "03",
                  title: "Channel weights",
                  desc: "Global average pool the gradients → one importance weight per channel.",
                  color: "#f59e0b",
                },
                {
                  step: "04",
                  title: "Weighted activation",
                  desc: "Sum (weight × feature map), apply ReLU, normalize to [0, 1], and resize to input length.",
                  color: "#ef4444",
                },
              ].map((s) => (
                <div key={s.step} className="flex gap-4">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5"
                    style={{
                      background: s.color + "20",
                      color: s.color,
                      border: `1px solid ${s.color}30`,
                    }}
                  >
                    {s.step}
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-0.5">
                      {s.title}
                    </h4>
                    <p className="text-xs text-slate-500 leading-relaxed">
                      {s.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Right — visual demo */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="space-y-4"
          >
            {/* ECG signal */}
            <div className="glass-card p-5 border border-indigo-500/15">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
                Raw ECG signal
              </p>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={signalWithHeat} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                    <XAxis dataKey="x" hide />
                    <YAxis hide />
                    <Line
                      type="monotone"
                      dataKey="y"
                      stroke="#6366f1"
                      strokeWidth={1.5}
                      dot={false}
                      animationDuration={2000}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Grad-CAM heatmap bar */}
            <div className="glass-card p-5 border border-amber-500/15">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
                Grad-CAM attention heatmap
              </p>
              <div className="h-20">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={signalWithHeat} margin={{ top: 2, right: 5, left: -30, bottom: 2 }}>
                    <defs>
                      <linearGradient id="heatGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="x" hide />
                    <YAxis hide domain={[0, 1]} />
                    <Area
                      type="monotone"
                      dataKey="heat"
                      stroke="#ef4444"
                      strokeWidth={1.5}
                      fill="url(#heatGrad)"
                      dot={false}
                      animationDuration={2000}
                    />
                    <ReferenceLine y={0.4} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center justify-between mt-2 text-[10px] text-slate-600">
                <span>Low attention</span>
                <span className="text-amber-600">— 0.4 threshold</span>
                <span>High attention (QRS)</span>
              </div>
            </div>

            {/* Overlaid ECG */}
            <div className="glass-card p-5 border border-red-500/15">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
                ECG + Grad-CAM overlay (highlighted regions = most influential)
              </p>
              <div className="h-36 relative">
                {/* Background heatmap shading */}
                <div
                  className="absolute inset-0 pointer-events-none"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent 20%, rgba(239,68,68,0.07) 38%, rgba(239,68,68,0.18) 46%, rgba(239,68,68,0.09) 58%, transparent 68%)",
                    borderRadius: 8,
                  }}
                />
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={signalWithHeat} margin={{ top: 5, right: 5, left: -30, bottom: 5 }}>
                    <XAxis dataKey="x" hide />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1220",
                        border: "1px solid rgba(239,68,68,0.3)",
                        borderRadius: 8,
                        fontSize: 11,
                        color: "#e2e8f0",
                      }}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      formatter={(v: any) => [
                        typeof v === "number" ? v.toFixed(3) : String(v ?? ""),
                        "Value",
                      ]}
                      labelFormatter={() => ""}
                    />
                    <Line
                      type="monotone"
                      dataKey="y"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      animationDuration={2000}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center gap-4 mt-3 text-xs text-slate-500">
                <div className="flex items-center gap-1.5">
                  <div className="w-4 h-1 bg-red-500 rounded" />
                  ECG signal
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-4 h-4 rounded bg-red-500/20 border border-red-500/30" />
                  High-attention region (QRS complex)
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
