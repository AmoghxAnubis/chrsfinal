"use client";

import { useState, useEffect } from 'react';
import { Code } from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';

const Spline = dynamic(
  () => import('@splinetool/react-spline'),
  { 
    ssr: false,
    loading: () => (
      <div className="fixed inset-0 flex items-center justify-center bg-black z-50">
        <div className="animate-pulse text-purple-400">
          Loading 3D environment...
        </div>
      </div>
    )
  }
);

export default function ApiDownloadPage() {
  const [email, setEmail] = useState('');
  const [purpose, setPurpose] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, purpose }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate API package');
      }

      const data = await response.json();
      setApiKey(data.apiKey);
      setPassword(data.password);
      setIsSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative w-full min-h-screen bg-black overflow-hidden">
      {/* 3D Spline Background with reduced brightness */}
      {mounted && (
        <div className="fixed inset-0 z-0" style={{
          transform: 'scale(3.4)',
          transformOrigin: 'center',
          filter: 'brightness(0.8) contrast(1)' // Reduced brightness
        }}>
          <Spline
            scene="https://prod.spline.design/Q2ySEkAE51u4pReL/scene.splinecode"
            onLoad={() => console.log('Spline loaded successfully')}
            onError={(e) => console.error('Spline error:', e)}
            style={{
              width: '100%',
              height: '100%',
              position: 'absolute',
              top: 0,
              left: 0
            }}
          />
          {/* Darker overlay to further reduce brightness */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/50 to-black/70" />
        </div>
      )}

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
        {/* Semi-transparent card */}
        <div className="w-full max-w-lg bg-gradient-to-br from-purple-900/20 via-purple-800/25 to-indigo-900/30 backdrop-blur-[2px] rounded-xl border border-purple-500/30 shadow-lg transition-all duration-500 p-8">
          <div className="flex items-center justify-center mb-8">
            <Code className="h-10 w-10 text-purple-300/90 mr-3" />
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-200/90 to-white/90">
              API Package Download
            </h1>
          </div>
          
          {!isSubmitted ? (
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block mb-2 text-sm font-medium text-purple-200/90">
                  Email Address
                </label>
                <input
                  type="email"
                  className="w-full px-4 py-3 bg-black/30 border border-purple-500/40 rounded-lg text-white/90 placeholder-purple-400/70 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent transition-all"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-sm font-medium text-purple-200/90">
                  What are you going to do with this API?
                </label>
                <textarea
                  className="w-full px-4 py-3 bg-black/30 border border-purple-500/40 rounded-lg text-white/90 placeholder-purple-400/70 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent transition-all min-h-[120px]"
                  value={purpose}
                  onChange={(e) => setPurpose(e.target.value)}
                  placeholder="Describe your project or use case..."
                  required
                />
              </div>
              
              <div className="flex justify-end">
                <button
                  type="submit"
                  className="px-6 py-3 bg-gradient-to-r from-purple-600/90 to-indigo-600/90 text-white/90 font-medium rounded-lg hover:from-purple-700/90 hover:to-indigo-700/90 transition-all duration-300 shadow-lg hover:shadow-purple-500/20 disabled:opacity-50 flex items-center"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white/90" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </>
                  ) : (
                    'Submit'
                  )}
                </button>
              </div>
              
              {error && (
                <div className="p-3 bg-red-900/30 border border-red-500/40 rounded-lg text-red-100/90 text-sm">
                  {error}
                </div>
              )}
            </form>
          ) : (
            <div className="space-y-6">
              <div className="p-4 bg-green-900/30 border border-green-500/40 rounded-lg text-green-100/90 text-center">
                Your API package is ready for download!
              </div>
              
              <div className="flex items-center justify-between p-4 bg-black/30 border border-purple-500/40 rounded-lg">
                <div>
                  <h3 className="font-medium text-purple-100/90">BackendAPI_Package.zip</h3>
                  <p className="text-sm text-purple-300/80">Contains all necessary backend code</p>
                </div>
                <Link
                  href={`/api/download?key=${apiKey}`}
                  className="px-4 py-2 bg-gradient-to-r from-green-600/90 to-emerald-600/90 text-white/90 rounded-lg hover:from-green-700/90 hover:to-emerald-700/90 transition-all duration-300 shadow hover:shadow-green-500/20 flex items-center"
                >
                  Download
                </Link>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-purple-100/90">Your API Credentials</h3>
                <div className="p-4 bg-black/30 border border-purple-500/40 rounded-lg space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-purple-200/90 mb-1">API Key</label>
                    <div className="p-2 bg-black/40 rounded text-sm font-mono text-purple-100/90 overflow-x-auto">
                      {apiKey}
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-purple-200/90 mb-1">Password</label>
                    <div className="p-2 bg-black/40 rounded text-sm font-mono text-purple-100/90">
                      {password}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-3 bg-yellow-900/30 border border-yellow-500/40 rounded-lg text-yellow-100/90 text-sm">
                <strong>Important:</strong> Keep these credentials secure. Do not share them publicly.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}