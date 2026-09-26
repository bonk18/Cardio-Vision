"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { STATS } from "@/lib/data";

export default function StatsSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section className="py-20 px-6 relative overflow-hidden">
      {/* BG accent */}
      <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/5 via-purple-600/5 to-indigo-600/5" />

      <div className="max-w-7xl mx-auto relative">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6"
        >
          {STATS.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.08 }}
              className="text-center group"
            >
              <div className="text-3xl md:text-4xl font-extrabold shimmer-text mb-1 group-hover:scale-110 transition-transform duration-300">
                {stat.value}
                {stat.suffix && (
                  <span className="text-xl font-bold ml-0.5 text-slate-400">
                    {stat.suffix}
                  </span>
                )}
              </div>
              <div className="text-xs font-semibold text-slate-300 mb-1">
                {stat.label}
              </div>
              <div className="text-[10px] text-slate-600 leading-snug">
                {stat.description}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
