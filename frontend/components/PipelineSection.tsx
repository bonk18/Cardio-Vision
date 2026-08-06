"use client";

import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { PIPELINE_STEPS } from "@/lib/data";
import { Info } from "lucide-react";

export default function PipelineSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const [activeStep, setActiveStep] = useState<number | null>(null);

  return (
    <section id="pipeline" className="py-28 px-6 bg-[#080c18]/60">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-16"
        >
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-widest mb-4 block">
            How it works
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-5 tracking-tight">
            The <span className="gradient-text">7-step pipeline</span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
            From raw physiological signal to downloadable clinical report.
            Click any step to see implementation details.
          </p>
        </motion.div>

        {/* Desktop: horizontal timeline */}
        <div className="hidden lg:block">
          {/* Step nodes */}
          <div className="relative flex items-start justify-between mb-0">
            {/* Connecting line */}
            <div className="absolute top-8 left-[4%] right-[4%] h-px bg-gradient-to-r from-indigo-500/0 via-indigo-500/40 to-indigo-500/0" />

            {PIPELINE_STEPS.map((step, i) => {
              const isActive = activeStep === step.step;
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{ delay: i * 0.1 }}
                  className="flex flex-col items-center w-[13%] relative cursor-pointer"
                  onClick={() =>
                    setActiveStep(isActive ? null : step.step)
                  }
                >
                  {/* Step circle */}
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    className={`relative w-16 h-16 rounded-2xl flex items-center justify-center text-2xl z-10 transition-all duration-300 ${
                      isActive
                        ? "shadow-lg"
                        : "hover:scale-105"
                    }`}
                    style={{
                      background: isActive
                        ? `linear-gradient(135deg, ${step.color}30, ${step.color}15)`
                        : "rgba(13,18,32,0.8)",
                      border: `1px solid ${isActive ? step.color + "60" : "rgba(99,102,241,0.2)"}`,
                      boxShadow: isActive
                        ? `0 0 20px ${step.color}30`
                        : "none",
                    }}
                  >
                    {step.icon}
                    {isActive && (
                      <div
                        className="absolute inset-0 rounded-2xl animate-ping opacity-20"
                        style={{ background: step.color }}
                      />
                    )}
                  </motion.div>

                  {/* Step number */}
                  <div
                    className="mt-3 text-xs font-bold"
                    style={{ color: step.color }}
                  >
                    0{step.step}
                  </div>
                  <p className="text-center text-xs font-semibold text-slate-300 mt-1 leading-tight px-1">
                    {step.title}
                  </p>
                  <p className="text-center text-[10px] text-slate-500 mt-0.5">
                    {step.subtitle}
                  </p>
                </motion.div>
              );
            })}
          </div>

          {/* Detail panel */}
          <div className="mt-8 min-h-[140px]">
            {activeStep !== null && (() => {
              const step = PIPELINE_STEPS.find((s) => s.step === activeStep)!;
              return (
                <motion.div
                  key={activeStep}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-6 border"
                  style={{ borderColor: step.color + "30" }}
                >
                  <div className="flex items-start gap-4">
                    <div
                      className="text-3xl w-14 h-14 flex items-center justify-center rounded-xl flex-shrink-0"
                      style={{
                        background: step.color + "15",
                        border: `1px solid ${step.color}30`,
                      }}
                    >
                      {step.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-white mb-1">
                        Step {step.step} — {step.title}
                      </h3>
                      <p className="text-slate-300 text-sm leading-relaxed mb-3">
                        {step.description}
                      </p>
                      <div className="flex items-start gap-2 text-xs text-slate-500 bg-slate-900/50 rounded-lg px-3 py-2">
                        <Info size={12} className="mt-0.5 flex-shrink-0 text-indigo-400" />
                        <span>{step.detail}</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })()}
            {activeStep === null && (
              <div className="text-center text-slate-600 text-sm py-8">
                Click any step above for implementation details
              </div>
            )}
          </div>
        </div>

        {/* Mobile: vertical list */}
        <div className="lg:hidden space-y-4">
          {PIPELINE_STEPS.map((step, i) => (
            <motion.div
              key={step.step}
              initial={{ opacity: 0, x: -20 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ delay: i * 0.08 }}
              className="glass-card p-5 cursor-pointer"
              style={{
                border: `1px solid ${step.color}25`,
              }}
              onClick={() =>
                setActiveStep(activeStep === step.step ? null : step.step)
              }
            >
              <div className="flex items-center gap-4">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center text-xl flex-shrink-0"
                  style={{ background: step.color + "15" }}
                >
                  {step.icon}
                </div>
                <div>
                  <span
                    className="text-xs font-bold"
                    style={{ color: step.color }}
                  >
                    Step {step.step}
                  </span>
                  <h3 className="text-sm font-semibold text-white">
                    {step.title}
                  </h3>
                  <p className="text-xs text-slate-500">{step.subtitle}</p>
                </div>
              </div>
              {activeStep === step.step && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="mt-4 pt-4 border-t border-white/5 text-sm text-slate-400 leading-relaxed"
                >
                  <p className="mb-2">{step.description}</p>
                  <p className="text-xs text-slate-500 bg-slate-900/50 rounded p-2">
                    {step.detail}
                  </p>
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
