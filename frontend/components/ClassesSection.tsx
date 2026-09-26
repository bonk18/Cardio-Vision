"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { ARRHYTHMIA_CLASSES, ECG_WAVEFORM_DATA } from "@/lib/data";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";

// Generate class-specific demo waveforms
function getWaveformForClass(classId: number) {
  return ECG_WAVEFORM_DATA.map((d, i) => {
    const t = i / 360;
    let y = d.y;
    if (classId === 1) {
      // AFib: irregular baseline, no clear P-wave, fibrillatory baseline
      y += 0.12 * Math.sin(i * 0.3 + 1.2) * Math.random() + 0.05 * Math.random();
    } else if (classId === 2) {
      // PVC: wide QRS, no P-wave, bizarre morphology
      const pvc = 1.8 * Math.exp(-((t - 0.28) ** 2) / (2 * 0.018 ** 2));
      const pvcS = -0.6 * Math.exp(-((t - 0.34) ** 2) / (2 * 0.015 ** 2));
      if (t > 0.15 && t < 0.55) y = pvc + pvcS + (Math.random() - 0.5) * 0.04;
    } else if (classId === 3) {
      // Paced: sharp spike followed by wide paced beat
      const spike = 3.0 * Math.exp(-((t - 0.24) ** 2) / (2 * 0.003 ** 2));
      const pacedBeat = 0.6 * Math.exp(-((t - 0.30) ** 2) / (2 * 0.025 ** 2));
      if (t > 0.18 && t < 0.5) y = spike + pacedBeat + (Math.random() - 0.5) * 0.03;
    }
    return { x: i, y: parseFloat(y.toFixed(4)) };
  });
}

export default function ClassesSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="classes" className="py-28 px-6 bg-[#080c18]/60">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-16"
        >
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest mb-4 block">
            Classification Target
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-5 tracking-tight">
            4 arrhythmia <span className="gradient-text">classes</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
            CardioVision maps the 19 MIT-BIH annotation codes into 4 clinically
            meaningful categories using the AAMI EC57 standard grouping.
          </p>
        </motion.div>

        {/* Class cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-16">
          {ARRHYTHMIA_CLASSES.map((cls, i) => {
            const waveData = getWaveformForClass(cls.id);
            return (
              <motion.div
                key={cls.id}
                initial={{ opacity: 0, y: 30 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: i * 0.12 }}
                className="glass-card p-6 group hover:-translate-y-1 transition-all duration-300 hover:shadow-xl"
                style={{
                  border: `1px solid ${cls.borderColor}`,
                }}
              >
                {/* Card header */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-3 mb-1">
                      <span className="text-2xl">{cls.icon}</span>
                      <div>
                        <h3 className="text-xl font-bold text-white">
                          {cls.name}
                        </h3>
                        <p className="text-xs text-slate-500">{cls.short}</p>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className="text-xl font-bold"
                      style={{ color: cls.color }}
                    >
                      {cls.prevalence}
                    </div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">
                      of dataset
                    </div>
                  </div>
                </div>

                {/* Mini ECG */}
                <div className="h-24 -mx-2 mb-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={waveData.slice(0, 200)} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                      <XAxis dataKey="x" hide />
                      <YAxis hide domain={["auto", "auto"]} />
                      <Line
                        type="monotone"
                        dataKey="y"
                        stroke={cls.color}
                        strokeWidth={1.5}
                        dot={false}
                        animationDuration={1500}
                        animationEasing="ease-out"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Description */}
                <p className="text-sm text-slate-400 leading-relaxed mb-3">
                  {cls.description}
                </p>

                {/* Clinical note */}
                <p className="text-xs text-slate-500 italic border-l-2 pl-3"
                   style={{ borderColor: cls.color + "60" }}>
                  {cls.clinicalNote}
                </p>

                {/* Annotation tags */}
                <div className="mt-4 flex flex-wrap gap-1.5">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider mr-1">
                    MIT-BIH codes:
                  </span>
                  {cls.annotations.map((ann) => (
                    <span
                      key={ann}
                      className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold"
                      style={{
                        background: cls.bgColor,
                        color: cls.color,
                        border: `1px solid ${cls.borderColor}`,
                      }}
                    >
                      {ann}
                    </span>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* AAMI mapping table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.5 }}
          className="glass-card p-6 border border-indigo-500/10 overflow-x-auto"
        >
          <h3 className="text-base font-semibold text-white mb-4">
            Complete Annotation Map
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5">
                <th className="text-left text-xs text-slate-500 uppercase tracking-wider pb-3 pr-4">Class</th>
                <th className="text-left text-xs text-slate-500 uppercase tracking-wider pb-3 pr-4">MIT-BIH Symbols</th>
                <th className="text-left text-xs text-slate-500 uppercase tracking-wider pb-3">Clinical Description</th>
              </tr>
            </thead>
            <tbody>
              {ARRHYTHMIA_CLASSES.map((cls) => (
                <tr key={cls.id} className="border-b border-white/5 last:border-0">
                  <td className="py-3 pr-4">
                    <span
                      className="px-3 py-1 rounded-full text-xs font-bold"
                      style={{ background: cls.bgColor, color: cls.color }}
                    >
                      {cls.name}
                    </span>
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs text-slate-400">
                    {cls.annotations.join(", ")}
                  </td>
                  <td className="py-3 text-xs text-slate-400 leading-relaxed">
                    {cls.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </div>
    </section>
  );
}
