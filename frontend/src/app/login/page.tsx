"use client";

import Button from "../../components/ui/button";
import { Lock, Mail } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/auth-context";
import Spline from '@splinetool/react-spline';

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const { login, isLoading: authLoading } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email || !password) {
      toast.error("Please fill in all fields");
      return;
    }

    setIsLoading(true);
    
    try {
      await login(email, password);
      toast.success("Login successful!");
      router.push("/");
    } catch (err) {
      toast.error("Login failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Spline 3D Background */}
      <div className="absolute inset-0 z-0">
        <Spline 
          scene="https://prod.spline.design/9Rl5gAvSldKkw6rw/scene.splinecode"
          className="w-full h-full"
        />
      </div>
      
      {/* Dark Overlay with reduced opacity */}
      <div className="absolute inset-0 bg-[#0f0c29]/40 z-1" />

      {/* Login Form */}
      <div className="relative z-10 min-h-screen flex items-center justify-center">
        <div className="w-full max-w-md px-4">
          {/* More transparent card with glass effect */}
          <div className="bg-[#1e1b4b]/5 backdrop-blur-lg rounded-xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300">
            <h2 className="text-3xl font-bold text-white mb-6 text-center [text-shadow:_0_0_10px_rgba(159,122,234,0.7)]">
              Welcome
            </h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-purple-100 mb-1">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-purple-300" />
                  </div>
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full bg-[#0f0c29]/40 border border-purple-500/20 rounded-md pl-10 pr-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent"
                    placeholder="your@email.com"
                    required
                    disabled={isLoading || authLoading}
                  />
                </div>
              </div>
              
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-purple-100 mb-1">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-purple-300" />
                  </div>
                  <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full bg-[#0f0c29]/40 border border-purple-500/20 rounded-md pl-10 pr-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-transparent"
                    placeholder="••••••••"
                    required
                    disabled={isLoading || authLoading}
                  />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-purple-300 rounded"
                    disabled={isLoading || authLoading}
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-purple-100">
                    Remember me
                  </label>
                </div>
                
                <div className="text-sm">
                  <Link 
                    href="/forgot-password" 
                    className="font-medium text-purple-300 hover:text-purple-200"
                  >
                    Forgot password?
                  </Link>
                </div>
              </div>
              
              <div>
                <Button
                  type="submit"
                  disabled={isLoading || authLoading}
                  className="w-full bg-purple-600/80 hover:bg-purple-700/80 text-white font-bold py-3 px-6 rounded-lg transition-all duration-300 hover:scale-[1.02] flex items-center justify-center"
                >
                  {(isLoading || authLoading) ? (
                    <>
                      <span className="animate-spin mr-2">🌀</span>
                      Signing in...
                    </>
                  ) : (
                    "Sign in"
                  )}
                </Button>
              </div>
            </form>
            
            <div className="mt-6 text-center text-sm text-purple-200">
              Don't have an account?{' '}
              <Link 
                href="/signup" 
                className="font-medium text-purple-300 hover:text-purple-200 hover:underline transition-colors duration-200"
              >
                Create account
              </Link>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}