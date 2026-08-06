"use client";

import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";
import { Database, Brain, Eye, FileText, Cpu, Activity } from "lucide-react";

const features = [
  {
    icon: <Database size={22} />,
    title: "Multi-Format Data Support",
    description:
      "Ingests the full 48-record MIT-BIH Arrhythmia Database directly from PhysioNet via WFDB, plus custom CSV and EDF files from clinical devices.",
    color: "text-indigo-400",
    bg: "bg-indigo-500/10",
    border: "border-indigo-500/20",
  },
  {
    icon: <Activity size={22} />,
    title: "Clinical Signal Auditing",
    description:
      "Rule-based SignalAuditor checks for missing data, flat-line segments, amplitude outliers, and sampling rate mismatches — with PASS/WARNING/CRITICAL severity.",
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
  },
  {
    icon: <Brain size={22} />,
    title: "4 Model Architectures",
    description:
      "Train Hybrid CNN-LSTM, Self-Attention Hybrid, ResNet Hybrid, or SE-CNN Hybrid — all with the same BiLSTM temporal encoder and FC classifier head.",
    color: "text-violet-400",
    bg: "bg-violet-500/10",
    border: "border-violet-500/20",
  },
  {
    icon: <Eye size={22} />,
    title: "Grad-CAM Explainability",
    description:
      "1D Grad-CAM hooks into the final Conv layer to produce per-sample heatmaps showing which ECG regions (P, QRS, T) drove each classification decision.",
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
  },
  {
    icon: <FileText size={22} />,
    title: "Downloadable Clinical Reports",
    description:
      "Self-contained HTML reports with embedded Grad-CAM plots, audit summaries, probability bars, and model metrics. Ideal for clinician review.",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
  },
  {
    icon: <Cpu size={22} />,
    title: "CPU + CUDA Accelerated",
    description:
      "Runs fully on CPU for development and demo. CUDA-enabled for production training on GPUs like RTX 4060 — device auto-detected via PyTorch.",
    color: "text-sky-400",
    bg: "bg-sky-500/10",
    border: "border-sky-500/20",
  },
];

function FeatureCard({
  feature,
  index,
}: {
  feature: (typeof features)[0];
  index: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className={`glass-card p-6 border ${feature.border} hover:border-opacity-50 group transition-all duration-300 hover:-translate-y-1 hover:shadow-xl`}
    >
      <div className={`inline-flex p-2.5 rounded-xl ${feature.bg} ${feature.color} mb-4 group-hover:scale-110 transition-transform duration-300`}>
        {feature.icon}
      </div>
      <h3 className="text-base font-semibold text-white mb-2">{feature.title}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">{feature.description}</p>
    </motion.div>
  );
}

export default function OverviewSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="overview" className="py-28 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest mb-4 block">
            What is CardioVision?
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-5 tracking-tight">
            End-to-end ECG <span className="gradient-text">intelligence</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
            CardioVision is a research-grade clinical pipeline that takes raw ECG recordings 
            all the way through auditing, preprocessing, deep learning classification, and 
            explainability — in one unified system.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <FeatureCard key={f.title} feature={f} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
