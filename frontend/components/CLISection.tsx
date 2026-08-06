"use client";

import { useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import { Copy, Check, Terminal } from "lucide-react";

const commands = [
  {
    label: "Download MIT-BIH",
    cmd: "python main.py download",
    desc: "Downloads all 48 records from PhysioNet via WFDB",
    color: "#6366f1",
  },
  {
    label: "Preprocess",
    cmd: "python main.py preprocess",
    desc: "Audits, denoises, segments beats → saves .npz",
    color: "#8b5cf6",
  },
  {
    label: "Train Hybrid",
    cmd: "python main.py train --arch hybrid --epochs 50 --batch-size 128 --lr 0.001",
    desc: "Trains with early stopping, saves best checkpoint",
    color: "#a855f7",
  },
  {
    label: "Evaluate",
    cmd: "python main.py evaluate --arch hybrid",
    desc: "Runs test-set evaluation + full classification report",
    color: "#c084fc",
  },
  {
    label: "Generate Reports",
    cmd: "python main.py report --arch hybrid",
    desc: "Creates HTML clinical reports with Grad-CAM for all classes",
    color: "#10b981",
  },
  {
    label: "Run Tests",
    cmd: "python main.py test",
    desc: "Runs pytest test suite for codebase integrity",
    color: "#f59e0b",
  },
  {
    label: "Launch Dashboard",
    cmd: "streamlit run app.py",
    desc: "Opens interactive browser dashboard on localhost:8501",
    color: "#ef4444",
  },
];

function CommandCard({ cmd, index }: { cmd: (typeof commands)[0]; index: number }) {
  const [copied, setCopied] = useState(false);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  const copy = () => {
    navigator.clipboard.writeText(cmd.cmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: -20 }}
      animate={inView ? { opacity: 1, x: 0 } : {}}
      transition={{ delay: index * 0.07 }}
      className="glass-card p-4 group border border-white/5 hover:border-white/10 transition-all duration-300"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span
              className="text-[10px] font-bold px-2 py-0.5 rounded-full"
              style={{
                background: cmd.color + "20",
                color: cmd.color,
              }}
            >
              {cmd.label}
            </span>
          </div>
          <code className="block text-xs text-emerald-400 font-mono bg-black/30 rounded-lg px-3 py-2 break-all">
            {cmd.cmd}
          </code>
          <p className="text-xs text-slate-500 mt-2">{cmd.desc}</p>
        </div>
        <button
          onClick={copy}
          className="flex-shrink-0 p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-500 hover:text-white mt-1"
          title="Copy command"
        >
          {copied ? (
            <Check size={14} className="text-emerald-400" />
          ) : (
            <Copy size={14} />
          )}
        </button>
      </div>
    </motion.div>
  );
}

export default function CLISection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="cli" className="py-28 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-16 items-start">
          {/* Left */}
          <motion.div
            ref={ref}
            initial={{ opacity: 0, x: -40 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7 }}
          >
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-widest mb-4 block">
              Command-Line Interface
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 tracking-tight">
              Full pipeline{" "}
              <span className="gradient-text">from the terminal</span>
            </h2>
            <p className="text-slate-400 text-base leading-relaxed mb-8">
              <code className="text-sm text-indigo-300 font-mono bg-indigo-500/10 px-1.5 py-0.5 rounded">
                main.py
              </code>{" "}
              is a multi-functional CLI for automating the entire
              CardioVision workflow — ideal for server training, batch
              processing, and CI integration.
            </p>

            {/* Architecture choices */}
            <div className="glass-card p-5 border border-indigo-500/15 mb-6">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
                --arch options
              </p>
              <div className="grid grid-cols-2 gap-2">
                {["hybrid", "attention", "resnet", "se"].map((arch) => (
                  <div
                    key={arch}
                    className="flex items-center gap-2 text-xs text-slate-300 bg-black/20 rounded-lg px-3 py-2"
                  >
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                    <code className="font-mono">{arch}</code>
                  </div>
                ))}
              </div>
            </div>

            {/* Terminal icon */}
            <div className="flex items-center gap-3 text-slate-500">
              <Terminal size={16} />
              <span className="text-sm">
                Also supports{" "}
                <code className="text-indigo-400 font-mono text-xs">
                  pytest tests/ -v
                </code>{" "}
                directly for test isolation.
              </span>
            </div>
          </motion.div>

          {/* Right — command list */}
          <div className="space-y-3">
            {commands.map((cmd, i) => (
              <CommandCard key={cmd.label} cmd={cmd} index={i} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
