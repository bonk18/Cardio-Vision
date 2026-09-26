import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "CardioVision — AI-Powered Arrhythmia Detection",
  description:
    "An explainable deep learning platform for clinical ECG analysis and cardiac arrhythmia detection using a Hybrid CNN-LSTM architecture with Grad-CAM.",
  keywords: ["ECG", "arrhythmia", "deep learning", "CNN-LSTM", "Grad-CAM", "cardiology", "AI"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased`}>{children}</body>
    </html>
  );
}
