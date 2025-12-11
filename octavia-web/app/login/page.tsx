// app/login/page.tsx - UPDATED FOR SUPABASE
"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Mail, Lock, Loader2 } from "lucide-react";
import { useState } from "react";
import { useUser } from "@/contexts/UserContext";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useUser();
  const [form, setForm] = useState({
    email: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showResendLink, setShowResendLink] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setShowResendLink(false);
    
    if (!form.email.trim()) {
      setError('Please enter your email');
      return;
    }
    
    if (!form.email.includes('@')) {
      setError('Please enter a valid email address');
      return;
    }
    
    if (!form.password) {
      setError('Please enter your password');
      return;
    }

    setLoading(true);
    
    try {
      const result = await login(form.email, form.password);
      
      if (result.success) {
        router.push('/dashboard');
      } else {
        if (result.requiresVerification) {
          setError(result.error || 'Please verify your email before logging in.');
          setShowResendLink(true);
        } else {
          setError(result.error || 'Login failed. Please check your credentials.');
        }
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/auth/resend-verification`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: form.email }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setError('Verification email resent! Please check your inbox.');
      } else {
        setError(data.detail || 'Failed to resend verification email.');
      }
    } catch (err) {
      setError('Failed to resend verification email.');
      console.error('Resend error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center relative overflow-hidden">
      {/* Ambient Background Glows */}
      <div className="glow-purple-strong"
        style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
      <div className="glow-purple"
        style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

      <div className="relative z-10 w-full max-w-md p-6">
        <Link href="/" className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-8"
        >
          <div className="text-center mb-8">
            <div className="w-12 h-12 mx-auto mb-4 relative flex items-center justify-center">
              <img
                src="/lunartech_logo_small.png"
                alt="LunarTech Logo"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-white/30 blur-xl rounded-full opacity-20" />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Welcome Back</h1>
            <p className="text-slate-400 text-sm">Sign in to your Octavia account</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
              {showResendLink && (
                <button
                  onClick={handleResendVerification}
                  disabled={loading}
                  className="mt-2 text-sm text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50"
                >
                  Resend verification email
                </button>
              )}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  className="glass-input w-full !pl-12"
                  placeholder="name@example.com"
                  required
                  disabled={loading}
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="password"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  className="glass-input w-full !pl-12"
                  placeholder="••••••••"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="remember"
                  className="h-4 w-4 rounded border-white/10 bg-white/5 text-primary-purple focus:ring-primary-purple"
                />
                <label htmlFor="remember" className="ml-2 text-sm text-slate-400">
                  Remember me
                </label>
              </div>
              <Link href="/forgot-password" className="text-sm text-primary-purple-bright hover:text-white transition-colors">
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-border-beam mt-4 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="btn-border-beam-inner justify-center py-3">
                {loading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Signing in...
                  </div>
                ) : (
                  "Sign In"
                )}
              </div>
            </button>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-white/10"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-[#0D0221] px-2 text-slate-500">Or continue with</span>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3">
              <button 
                className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed" 
                disabled={loading}
                onClick={() => {
                  setError('Social login coming soon! Please use email login for now.');
                }}
              >
                <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                  <path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.74 1.18 0 2.21-.93 3.69-.93.95 0 2.58.5 3.63 1.62-3.28 1.66-2.57 6.62 1.3 8.21-.63 1.72-1.62 3.45-3.7 3.33zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
                </svg>
                Apple
              </button>
              <button 
                className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed" 
                disabled={loading}
                onClick={() => {
                  setError('Social login coming soon! Please use email login for now.');
                }}
              >
                <span className="font-bold">G</span>
                Google
              </button>
            </div>
          </div>

          <p className="mt-8 text-center text-sm text-slate-400">
            Don't have an account?{" "}
            <Link href="/signup" className="text-primary-purple-bright hover:text-white transition-colors font-medium">
              Sign up
            </Link>
          </p>
          
          <div className="mt-4 text-center">
            <Link 
              href="/verify-email" 
              className="inline-block text-sm text-slate-500 hover:text-slate-300 transition-colors"
            >
              Need to verify your email? Click here
            </Link>
          </div>
          
          {/* Demo Account Button */}
          <div className="mt-6 pt-4 border-t border-white/10">
            <button
              onClick={async () => {
                setLoading(true);
                try {
                  // Use demo login endpoint
                  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/auth/demo-login`, {
                    method: 'POST',
                  });
                  let data = null;
                  try {
                    data = await response.json();
                  } catch (jsonErr) {
                    data = null;
                  }
                  if (data && data.success) {
                    // Store user data
                    localStorage.setItem('octavia_user', JSON.stringify({
                      ...data.user,
                      token: data.token
                    }));
                    router.push('/dashboard');
                  } else if (data && data.detail) {
                    setError(data.detail || 'Demo login failed');
                  } else {
                    setError('Demo login failed: No response from server');
                  }
                } catch (err) {
                  setError('Failed to login with demo account');
                  console.error('Demo login error:', err);
                } finally {
                  setLoading(false);
                }
              }}
              disabled={loading}
              className="w-full py-2 rounded-lg bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition-colors text-purple-300 text-sm disabled:opacity-50"
            >
              Try Demo Account
            </button>
            <p className="text-xs text-center text-slate-500 mt-2">
              Use our demo account to explore features
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}