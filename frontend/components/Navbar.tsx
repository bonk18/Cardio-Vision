"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Heart, Menu, X, ExternalLink } from "lucide-react";

const navLinks = [
  { label: "Overview", href: "#overview" },
  { label: "Pipeline", href: "#pipeline" },
  { label: "Architecture", href: "#architecture" },
  { label: "Classes", href: "#classes" },
  { label: "Tech Stack", href: "#tech" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 30);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-[#060912]/90 backdrop-blur-xl border-b border-indigo-500/10 shadow-2xl"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="flex items-center gap-3 group">
          <div className="relative">
            <Heart
              className="text-indigo-400 group-hover:text-indigo-300 transition-colors animate-beat"
              size={22}
              fill="currentColor"
            />
            <div
              className="absolute inset-0 rounded-full bg-indigo-500/20"
              style={{
                animation: "pulse-ring 1.5s ease-out infinite",
              }}
            />
          </div>
          <span className="font-bold text-lg tracking-tight text-white">
            Cardio<span className="gradient-text">Vision</span>
          </span>
        </a>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-all duration-200"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* CTA */}
        <div className="hidden md:flex items-center gap-3">
          <a
            href="/app"
            className="flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:shadow-lg hover:shadow-indigo-500/25 transition-all duration-300 hover:-translate-y-0.5"
          >
            Launch App
            <ExternalLink size={13} />
          </a>
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden text-slate-400 hover:text-white transition-colors"
          onClick={() => setMobileOpen((v) => !v)}
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-[#0d1220]/95 backdrop-blur-xl border-t border-indigo-500/10 px-6 pb-6"
          >
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                className="block py-3 text-sm text-slate-400 hover:text-white border-b border-white/5 transition-colors"
              >
                {link.label}
              </a>
            ))}
            <a
              href="/app"
              className="flex items-center justify-center gap-2 w-full py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-500 to-purple-600 text-white"
            >
              Launch App <ExternalLink size={13} />
            </a>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
