"use client";
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import Spline from '@splinetool/react-spline';
import { useAuth } from "@/context/auth-context";

export default function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await axios.post(
        'http://localhost:5000/api/signup',
        { email, password },
        {
          withCredentials: true,
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );

      if (response.data.success) {
        // Call the login function from auth context
        await login(email, password);
        router.push('/');
      }
    } catch (error: any) {
      if (error.response) {
        setError(error.response.data.error || 'Signup failed');
      } else if (error.request) {
        setError('Network error. Please try again.');
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Spline 3D Background */}
      <div className="fixed inset-0 -z-10 h-full w-full">
        <Spline 
          scene="https://prod.spline.design/9Rl5gAvSldKkw6rw/scene.splinecode"
          className="w-full h-full object-cover"
        />
      </div>

      {/* Signup Form */}
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="max-w-md w-full p-8 rounded-2xl backdrop-blur-lg bg-white/10 border border-white/20 shadow-lg">
          <h1 className="text-3xl font-bold mb-6 text-center text-white">Sign Up</h1>
          
          {error && (
            <div className="mb-4 p-3 bg-red-500/20 rounded-lg">
              <p className="text-red-300 text-center">{error}</p>
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label htmlFor="email" className="block text-sm font-medium mb-2 text-white/80">
                Email
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white/10 text-white placeholder-white/50 border border-white/20"
                required
                placeholder="your@email.com"
                disabled={isLoading}
              />
            </div>
            <div className="mb-6">
              <label htmlFor="password" className="block text-sm font-medium mb-2 text-white/80">
                Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white/10 text-white placeholder-white/50 border border-white/20"
                required
                minLength={8}
                placeholder="••••••••"
                disabled={isLoading}
              />
            </div>
            <button
              type="submit"
              className="w-full bg-purple-500/90 hover:bg-purple-600 text-white py-3 px-4 rounded-lg transition duration-200 focus:outline-none focus:ring-2 focus:ring-white font-medium disabled:opacity-70 disabled:cursor-not-allowed"
              disabled={isLoading}
            >
              {isLoading ? 'Creating account...' : 'Create Account'}
            </button>
          </form>
          <p className="mt-6 text-center text-sm text-white/70">
            Already have an account?{' '}
            <a href="/login" className="text-white font-medium hover:underline">
              Log in here
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}