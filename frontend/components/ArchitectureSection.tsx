"use client";

import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { MODEL_ARCHITECTURES } from "@/lib/data";
import { CheckCircle } from "lucide-react";

export default function ArchitectureSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const [selected, setSelected] = useState("hybrid");

  const arch = MODEL_ARCHITECTURES.find((a) => a.id === selected)!;

  return (
    <section id="architecture" className="py-28 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-16"
        >
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest mb-4 block">
            Model Architectures
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-5 tracking-tight">
            Four <span className="gradient-text">network variants</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
            All architectures share the same preprocessing pipeline, training loop,
            and evaluation — only the feature extraction backbone changes.
          </p>
        </motion.div>

        {/* Arch tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.2 }}
          className="flex flex-wrap justify-center gap-3 mb-12"
        >
          {MODEL_ARCHITECTURES.map((a) => (
            <button
              key={a.id}
              onClick={() => setSelected(a.id)}
              className={`relative px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
                selected === a.id
                  ? "text-white shadow-lg"
                  : "text-slate-400 hover:text-slate-200 glass-card hover:bg-white/5"
              }`}
              style={
                selected === a.id
                  ? {
                      background: `linear-gradient(135deg, ${a.badgeColor}30, ${a.badgeColor}15)`,
                      border: `1px solid ${a.badgeColor}50`,
                      boxShadow: `0 0 20px ${a.badgeColor}20`,
                    }
                  : {}
              }
            >
              {a.name}
              <span
                className="ml-2 text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                style={{
                  background: a.badgeColor + "20",
                  color: a.badgeColor,
                }}
              >
                {a.badge}
              </span>
            </button>
          ))}
        </motion.div>

        {/* Detail panel */}
        <div className="grid lg:grid-cols-5 gap-8">
          {/* Left: architecture layers */}
          <div className="lg:col-span-3">
            <motion.div
              key={selected}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
              className="glass-card p-6 h-full"
              style={{ border: `1px solid ${arch.badgeColor}20` }}
            >
              <h3 className="text-lg font-bold text-white mb-2">{arch.name}</h3>
              <p className="text-sm text-slate-400 mb-6 leading-relaxed">
                {arch.description}
              </p>

              {/* Layer flow */}
              <div className="space-y-2">
                {arch.layers.map((layer, i) => (
                  <motion.div
                    key={layer.name}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.07 }}
                    className="flex items-center gap-3"
                  >
                    {/* Connector line */}
                    <div className="flex flex-col items-center w-6 flex-shrink-0">
                      <div
                        className="w-6 h-6 rounded-lg flex items-center justify-center text-[10px] font-bold text-white"
                        style={{ background: layer.color }}
                      >
                        {i + 1}
                      </div>
                      {i < arch.layers.length - 1 && (
                        <div
                          className="w-px flex-1 min-h-[12px]"
                          style={{ background: layer.color + "40" }}
                        />
                      )}
                    </div>

                    {/* Layer info */}
                    <div
                      className="flex-1 flex items-center justify-between rounded-xl px-3 py-2.5 text-sm"
                      style={{
                        background: layer.color + "10",
                        border: `1px solid ${layer.color}25`,
                      }}
                    >
                      <span className="text-slate-200 font-medium text-xs">
                        {layer.name}
                      </span>
                      <span
                        className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md"
                        style={{
                          background: layer.color + "20",
                          color: layer.color,
                        }}
                      >
                        {layer.shape}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right: pros + params */}
          <div className="lg:col-span-2 space-y-6">
            {/* Strengths */}
            <motion.div
              key={selected + "-pros"}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="glass-card p-6"
              style={{ border: `1px solid ${arch.badgeColor}20` }}
            >
              <h4 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">
                Strengths
              </h4>
              <div className="space-y-3">
                {arch.pros.map((p) => (
                  <div key={p} className="flex items-center gap-3">
                    <CheckCircle
                      size={15}
                      style={{ color: arch.badgeColor }}
                      className="flex-shrink-0"
                    />
                    <span className="text-sm text-slate-300">{p}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Shared hyperparams */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.2 }}
              className="glass-card p-6 border border-indigo-500/10"
            >
              <h4 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">
                Shared Config
              </h4>
              {[
                { k: "Optimizer", v: "AdamW (lr=1e-3)" },
                { k: "Loss", v: "CrossEntropyLoss (label_smooth=0.1)" },
                { k: "LR Schedule", v: "ReduceLROnPlateau (×0.5)" },
                { k: "Early Stop", v: "Patience 15 epochs" },
                { k: "Batch Size", v: "128" },
                { k: "Sampling", v: "Weighted random (balanced)" },
              ].map((row) => (
                <div
                  key={row.k}
                  className="flex justify-between text-xs py-1.5 border-b border-white/5 last:border-0"
                >
                  <span className="text-slate-500">{row.k}</span>
                  <span className="text-slate-200 font-mono">{row.v}</span>
                </div>
              ))}
            </motion.div>

            {/* CLI badge */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card p-4 border border-emerald-500/15"
            >
              <p className="text-xs text-slate-500 mb-2">Train via CLI:</p>
              <code className="text-xs text-emerald-400 font-mono">
                python main.py train --arch {selected} --epochs 50
              </code>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
