"use client";

import { Heart, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="py-12 px-6 border-t border-white/5 bg-[#060912]">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-10">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Heart size={18} className="text-indigo-400 animate-beat" fill="currentColor" />
              <span className="font-bold text-white">
                Cardio<span className="gradient-text">Vision</span>
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed max-w-xs">
              An end-to-end explainable deep learning platform for clinical ECG
              arrhythmia detection.
            </p>
          </div>

          {/* Links */}
          <div>
            <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Resources
            </h5>
            <div className="space-y-2">
              {[
                { label: "Streamlit Dashboard", href: "/app", external: true },
                { label: "MIT-BIH on PhysioNet", href: "https://physionet.org/content/mitdb/1.0.0/", external: true },
                { label: "Pipeline Overview", href: "#pipeline" },
                { label: "Model Architectures", href: "#architecture" },
              ].map((l) => (
                <a
                  key={l.label}
                  href={l.href}
                  target={l.external ? "_blank" : undefined}
                  rel={l.external ? "noopener noreferrer" : undefined}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {l.label}
                  {l.external && <ExternalLink size={10} />}
                </a>
              ))}
            </div>
          </div>

          {/* Tech */}
          <div>
            <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Stack
            </h5>
            <div className="space-y-1.5">
              {[
                "PyTorch 2.x",
                "Streamlit 1.60",
                "WFDB + MIT-BIH",
                "SciPy Signal Processing",
                "Grad-CAM Explainability",
              ].map((t) => (
                <div key={t} className="text-xs text-slate-500">
                  {t}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="border-t border-white/5 pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-600">
            CardioVision — MIT License. For research and educational use.
            Clinical deployment requires qualified medical review.
          </p>
          <p className="text-xs text-slate-600">
            Powered by PyTorch · MIT-BIH Arrhythmia Database (PhysioNet)
          </p>
        </div>
      </div>
    </footer>
  );
}
