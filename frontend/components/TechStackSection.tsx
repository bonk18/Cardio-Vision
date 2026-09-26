"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { TECH_STACK } from "@/lib/data";

export default function TechStackSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="tech" className="py-28 px-6 bg-[#080c18]/60">
      <div className="max-w-7xl mx-auto">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-16"
        >
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest mb-4 block">
            Built With
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-5 tracking-tight">
            Technology <span className="gradient-text">stack</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
            Every component is purpose-selected for clinical ECG analysis —
            from signal processing to interactive visualization.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {TECH_STACK.map((tech, i) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: i * 0.06, duration: 0.4 }}
              whileHover={{ y: -4, scale: 1.03 }}
              className="glass-card p-5 flex flex-col items-center text-center cursor-default group transition-all duration-300 border border-white/5 hover:border-white/10"
            >
              <div className="text-3xl mb-3 group-hover:scale-110 transition-transform duration-300">
                {tech.icon}
              </div>
              <h4 className="text-sm font-bold text-white mb-1">{tech.name}</h4>
              <p className="text-[11px] text-slate-500 leading-snug">{tech.role}</p>
            </motion.div>
          ))}
        </div>

        {/* License + dataset credit */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.6 }}
          className="mt-12 grid md:grid-cols-2 gap-6"
        >
          <div className="glass-card p-6 border border-emerald-500/15">
            <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              🏥 Dataset — MIT-BIH Arrhythmia Database
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              The gold-standard reference database for cardiac arrhythmia
              research. 48 half-hour two-channel ECG recordings from 47 subjects
              sampled at 360 Hz, with expert annotations for 19 beat types.
              Hosted freely on{" "}
              <strong className="text-slate-300">PhysioNet</strong>.
            </p>
          </div>
          <div className="glass-card p-6 border border-indigo-500/15">
            <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
              📄 License — MIT
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              CardioVision is open-source under the{" "}
              <strong className="text-slate-300">MIT License</strong>. Free to
              use, modify, and distribute for research and educational purposes.
              Clinical deployment requires qualified medical oversight.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
