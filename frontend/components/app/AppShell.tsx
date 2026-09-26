"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard, Upload, Brain, Microscope, FileText,
  Heart, ChevronLeft, ChevronRight, Cpu, Zap, AlertCircle,
} from "lucide-react";
import type { StatusResponse, InferenceResponse } from "@/lib/api";
import { getStatus } from "@/lib/api";
import DashboardPage from "./pages/DashboardPage";
import UploadPage from "./pages/UploadPage";
import TrainPage from "./pages/TrainPage";
import InferencePage from "./pages/InferencePage";
import ReportsPage from "./pages/ReportsPage";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "upload",    label: "Upload & Preprocess", icon: Upload },
  { id: "train",     label: "Train Model", icon: Brain },
  { id: "infer",     label: "Inference & Explain", icon: Microscope },
  { id: "reports",   label: "Reports", icon: FileText },
];

const CLASS_COLORS: Record<string, string> = {
  Normal: "#10b981", AFib: "#f59e0b", PVC: "#ef4444", Global: "#3b82f6",
};

export default function AppShell() {
  const [page, setPage] = useState("dashboard");
  const [collapsed, setCollapsed] = useState(false);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [apiError, setApiError] = useState(false);

  // Shared cross-page state
  const [lastInference, setLastInference] = useState<import("@/lib/api").InferenceResponse | null>(null);
  const [selectedArch, setSelectedArch] = useState("hybrid");

  useEffect(() => {
    const poll = () =>
      getStatus()
        .then((s) => { setStatus(s); setApiError(false); })
        .catch(() => setApiError(true));
    poll();
    const id = setInterval(poll, 8000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex h-screen bg-[#060912] text-slate-200 overflow-hidden">
      {/* ── Sidebar ─────────────────────────────── */}
      <motion.aside
        animate={{ width: collapsed ? 72 : 260 }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="flex flex-col bg-[#0d1220] border-r border-indigo-500/10 flex-shrink-0 overflow-hidden"
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-indigo-500/10">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
            <Heart size={18} fill="white" className="text-white animate-beat" />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
              >
                <div className="font-bold text-sm text-white leading-tight">CardioVision</div>
                <div className="text-[10px] text-slate-500">Arrhythmia Detection</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Nav items */}
        <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
          {NAV.map(({ id, label, icon: Icon }) => {
            const active = page === id;
            return (
              <button
                key={id}
                onClick={() => setPage(id)}
                title={collapsed ? label : undefined}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                  active
                    ? "bg-indigo-500/15 text-indigo-300 border border-indigo-500/25"
                    : "text-slate-500 hover:text-slate-200 hover:bg-white/5"
                }`}
              >
                <Icon size={17} className="flex-shrink-0" />
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.1 }}
                      className="truncate"
                    >
                      {label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>
            );
          })}
        </nav>

        {/* Arch selector */}
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="px-3 pb-3"
            >
              <p className="text-[10px] text-slate-600 uppercase tracking-widest mb-2 px-1">
                Architecture
              </p>
              <select
                value={selectedArch}
                onChange={(e) => setSelectedArch(e.target.value)}
                className="w-full bg-[#060912] border border-indigo-500/20 text-slate-300 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500/40"
              >
                {["hybrid", "attention", "resnet", "se"].map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status footer */}
        <div className="px-3 pb-4 border-t border-indigo-500/10 pt-3">
          {apiError ? (
            <div className="flex items-center gap-2 text-red-400 text-xs px-1">
              <AlertCircle size={13} />
              {!collapsed && <span>API offline</span>}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs px-1">
              {status?.cuda ? (
                <Zap size={13} className="text-green-400 flex-shrink-0" />
              ) : (
                <Cpu size={13} className="text-yellow-400 flex-shrink-0" />
              )}
              <AnimatePresence>
                {!collapsed && status && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-slate-500 truncate"
                  >
                    {status.cuda ? status.gpu_name ?? "CUDA" : "CPU"}
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
          )}
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center justify-center h-10 border-t border-indigo-500/10 text-slate-600 hover:text-slate-300 transition-colors"
        >
          {collapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
        </button>
      </motion.aside>

      {/* ── Main content ────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="min-h-full"
          >
            {page === "dashboard" && (
              <DashboardPage status={status} apiError={apiError} selectedArch={selectedArch} classColors={CLASS_COLORS} onNavigate={setPage} />
            )}
            {page === "upload" && (
              <UploadPage status={status} />
            )}
            {page === "train" && (
              <TrainPage selectedArch={selectedArch} status={status} />
            )}
            {page === "infer" && (
              <InferencePage
                selectedArch={selectedArch}
                status={status}
                lastInference={lastInference as InferenceResponse | null}
                setLastInference={setLastInference as (r: InferenceResponse) => void}
              />
            )}
            {page === "reports" && (
              <ReportsPage lastInference={lastInference} />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
