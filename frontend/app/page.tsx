import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import StatsSection from "@/components/StatsSection";
import OverviewSection from "@/components/OverviewSection";
import PipelineSection from "@/components/PipelineSection";
import ArchitectureSection from "@/components/ArchitectureSection";
import ClassesSection from "@/components/ClassesSection";
import GradCamSection from "@/components/GradCamSection";
import CLISection from "@/components/CLISection";
import TechStackSection from "@/components/TechStackSection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <HeroSection />
      <StatsSection />
      <OverviewSection />
      <PipelineSection />
      <ArchitectureSection />
      <ClassesSection />
      <GradCamSection />
      <CLISection />
      <TechStackSection />
      <Footer />
    </main>
  );
}
