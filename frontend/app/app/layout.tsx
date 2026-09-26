import type { Metadata } from "next";
export const metadata: Metadata = { title: "CardioVision — Pipeline" };
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
