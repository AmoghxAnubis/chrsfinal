"use client";

import Button from "../components/ui/button";
import { BrainCircuit, HomeIcon, FlaskConical, Cpu, Database, LineChart, Code } from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/auth-context";
import Spline from '@splinetool/react-spline';

export default function Home() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [cursorInViewport, setCursorInViewport] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedMolecules, setGeneratedMolecules] = useState<any[]>([]);
  const [constraints, setConstraints] = useState<any>(null);
  const [splineLoaded, setSplineLoaded] = useState(false);
  const [description, setDescription] = useState("");
  const [apiReady, setApiReady] = useState(false);
  const router = useRouter();
  const { user, logout } = useAuth();

  // Mouse movement effects
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
      setCursorInViewport(true);
    };

    const handleMouseLeave = () => {
      setCursorInViewport(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseout', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseout', handleMouseLeave);
    };
  }, []);

  // Check API status on load
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/health');
        if (response.ok) {
          setApiReady(true);
          toast.success('API connected successfully');
        }
      } catch (error) {
        console.error('Error checking API status:', error);
        toast.error('Failed to connect to the drug discovery API');
      }
    };
    checkApiStatus();
  }, []);

  const generateMolecules = async () => {
    if (!description.trim()) {
      toast.error('Please enter a drug description');
      return;
    }

    if (!apiReady) {
      toast.error('API not ready. Please try again later.');
      return;
    }

    setIsGenerating(true);
    toast.info('Generating molecules... This may take a few minutes...');
    
    try {
      const response = await fetch('http://localhost:5000/api/generate-drugs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: description
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Generation failed');
      }
      
      const data = await response.json();
      setGeneratedMolecules(data.molecules);
      setConstraints(data.constraints);
      toast.success(`Generated ${data.molecules.length} potential drug candidates!`);
    } catch (error: any) {
      toast.error(`Generation failed: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Layered Background with Purple Gradient */}
      <div className="fixed inset-0 -z-50 h-screen w-full overflow-hidden">
        {/* Base Gradient Layer */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#2e1065] via-[#4c1d95] to-[#6d28d9] opacity-90" />
        
        {/* Spline 3D Model Container */}
        <div className={`absolute inset-0 transition-opacity duration-1000 ${splineLoaded ? 'opacity-100' : 'opacity-0'}`}>
          {/* Model with increased visibility */}
          <div className="absolute inset-0 scale-[3.3] origin-center">
            <Spline 
              scene="https://prod.spline.design/Q2ySEkAE51u4pReL/scene.splinecode"
              onLoad={() => setSplineLoaded(true)}
              className="w-full h-full mix-blend-lighten"
            />
          </div>
          
          {/* Purple Glow Effect */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(139,92,246,0.5)_0%,transparent_70%)] pointer-events-none" />
        </div>
        
        {/* Overlay for contrast */}
        <div className="absolute inset-0 bg-[#0f0c29]/20 backdrop-blur-[1px]" />
      </div>

      {/* Smoky Navbar */}
      <header className="fixed w-full z-50 group">
        <div className="absolute inset-x-0 top-0 h-20 bg-gradient-to-b from-[#3a1c6e]/70 to-transparent backdrop-blur-sm transition-all duration-300 group-hover:backdrop-blur-md group-hover:from-[#3a1c6e]/90" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center hover:[text-shadow:_0_0_15px_rgba(159,122,234,0.9)] transition-all duration-300">
              <BrainCircuit className="h-8 w-8 text-purple-300 mr-2 drop-shadow-[0_0_8px_rgba(159,122,234,0.8)] hover:drop-shadow-[0_0_15px_rgba(159,122,234,1)] transition-all duration-300" />
              <span className="text-xl font-bold text-purple-100 [text-shadow:_0_0_10px_rgba(159,122,234,0.7)]">
                CHRS
              </span>
            </div>

            <nav className="hidden md:flex items-center space-x-8 absolute left-1/2 transform -translate-x-1/2">
              <Link href="/" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                <HomeIcon className="h-4 w-4 mr-1 inline drop-shadow-[0_0_4px_rgba(159,122,234,0.6)] hover:drop-shadow-[0_0_8px_rgba(255,255,255,0.8)] transition-all duration-300" />
                Home
              </Link>
              <Link href="/product" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                Product
              </Link>
              <Link href="/api-download" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                <Code className="h-4 w-4 mr-1 inline" />
                API
              </Link>
              <Link href="/company" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                Company
              </Link>
            </nav>

            <Button 
              variant="outline" 
              className="border-purple-300 text-purple-100 hover:bg-[#3a1c6e]/40 hover:border-purple-400/60 transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.5)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]"
              onClick={() => user ? logout() : router.push('/login')}
            >
              {user ? (
                <>
                  <span className="mr-1">👤</span> Logout
                </>
              ) : (
                'Login'
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="relative z-10 min-h-screen flex items-center justify-center pt-28">
        <div className="text-center px-4 max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 text-white drop-shadow-lg hover:[text-shadow:_0_0_20px_rgba(255,255,255,0.3)] transition-all duration-500">
            Comprehensive Health Research System
          </h1>
          
          <p className="text-2xl text-purple-100 my-6 drop-shadow-md">
            with
          </p>
          
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-purple-100 mb-8 drop-shadow-lg hover:[text-shadow:_0_0_20px_rgba(159,122,234,0.8)] transition-all duration-500">
            AI-Powered Drug Discovery
          </h2>

          {/* Drug Description Input */}
          <div className="mt-8">
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter drug description (e.g., 'An anti-inflammatory drug for arthritis pain')"
              className="w-full max-w-2xl h-32 p-4 bg-[#1e1b4b]/50 backdrop-blur-sm text-white rounded-lg border border-purple-500/30 focus:border-purple-500/60 focus:outline-none resize-none"
            />
          </div>

          <div className="flex justify-center mt-8 relative">
            {/* Generate Molecules Button */}
            <div className="relative">
              <div className={`relative rounded-lg overflow-hidden transition-all ${
                isGenerating ? 'p-[4px]' : 'p-0'
              }`}>
                {isGenerating && (
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-400 via-indigo-500 to-indigo-600 animate-spin-slow" />
                )}
                <Button 
                  onClick={generateMolecules}
                  disabled={!user || !description.trim() || !apiReady || isGenerating}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-300 relative z-10 w-full"
                >
                  {isGenerating ? (
                    <span className="text-white">Generating...</span>
                  ) : (
                    <span className="flex items-center">
                      <Cpu className="mr-2" /> 
                      {!apiReady ? 'API Not Ready' : (user ? 'Generate Drug Candidates' : 'Login to Generate')}
                    </span>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {generatedMolecules.length > 0 && (
        <div className="relative z-10 bg-[#0f0c29]/90 backdrop-blur-sm py-16 px-4">
          <div className="max-w-6xl mx-auto">
            <h3 className="text-3xl font-bold text-white mb-8 text-center flex items-center justify-center">
              <Database className="mr-2" /> Generated Drug Candidates
            </h3>
            
            {/* Constraints Section */}
            {constraints && (
              <div className="mb-8 bg-[#1e1b4b]/70 backdrop-blur-sm rounded-xl p-6 border border-purple-500/30">
                <h4 className="text-xl font-semibold text-white mb-4">Generation Constraints</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  <div className="bg-[#3a1c6e]/50 p-3 rounded">
                    <div className="text-purple-200 text-sm">Molecular Weight</div>
                    <div className="text-white font-mono">{constraints.molecular_weight_range[0]} - {constraints.molecular_weight_range[1]}</div>
                  </div>
                  <div className="bg-[#3a1c6e]/50 p-3 rounded">
                    <div className="text-purple-200 text-sm">LogP</div>
                    <div className="text-white font-mono">{constraints.logp_range[0]} - {constraints.logp_range[1]}</div>
                  </div>
                  <div className="bg-[#3a1c6e]/50 p-3 rounded">
                    <div className="text-purple-200 text-sm">HBA</div>
                    <div className="text-white font-mono">{constraints.hba_range[0]} - {constraints.hba_range[1]}</div>
                  </div>
                  <div className="bg-[#3a1c6e]/50 p-3 rounded">
                    <div className="text-purple-200 text-sm">HBD</div>
                    <div className="text-white font-mono">{constraints.hbd_range[0]} - {constraints.hbd_range[1]}</div>
                  </div>
                  <div className="bg-[#3a1c6e]/50 p-3 rounded">
                    <div className="text-purple-200 text-sm">Rotatable Bonds</div>
                    <div className="text-white font-mono">≤ {constraints.rotatable_bonds}</div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {generatedMolecules.map((mol, index) => (
                <div key={index} className="bg-[#1e1b4b]/70 backdrop-blur-sm rounded-xl p-6 border border-purple-500/30 hover:border-purple-500/60 transition-colors duration-300">
                  <h4 className="text-xl font-semibold text-white mb-2">Candidate {index + 1}</h4>
                  <p className="text-purple-200 font-mono mb-4">{mol.smiles}</p>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm text-white">
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">MW:</span> {mol.molecular_weight.toFixed(1)}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">LogP:</span> {mol.logp.toFixed(2)}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">HBA:</span> {mol.hba}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">HBD:</span> {mol.hbd}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">Rot. Bonds:</span> {mol.rotatable_bonds}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">Rings:</span> {mol.rings}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">PSA:</span> {mol.psa.toFixed(1)}
                    </div>
                    <div className="bg-[#3a1c6e]/50 p-2 rounded">
                      <span className="text-purple-200">Aromatic:</span> {mol.aromatic_rings}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}