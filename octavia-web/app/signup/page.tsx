// app/signup/page.tsx - UPDATED WITH SUPABASE INTEGRATION
"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Mail, Lock, User, Loader2 } from "lucide-react";
import { useState } from "react";
import { useUser } from "@/contexts/UserContext";
import { useRouter } from "next/navigation";

export default function SignupPage() {
  const router = useRouter();
  const { signup } = useUser();
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Validation
    if (!form.name.trim()) {
      setError('Please enter your name');
      return;
    }

    if (!form.email.trim()) {
      setError('Please enter your email');
      return;
    }

    if (!form.email.includes('@')) {
      setError('Please enter a valid email address');
      return;
    }

    if (form.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const result = await signup(form.email, form.password, form.name);

      if (result.success) {
        if (result.requiresVerification) {
          setSuccess(result.message || 'Verification email sent! Please check your inbox.');
          // Clear form
          setForm({
            name: '',
            email: '',
            password: '',
            confirmPassword: '',
          });

          // Option 1: Show success message and link to verification page
          setTimeout(() => {
            router.push('/verify-email');
          }, 2000);
        } else {
          // This shouldn't happen with Supabase, but handle it just in case
          router.push('/dashboard');
        }
      } else {
        setError(result.error || 'Signup failed. Please try again.');
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
      console.error('Signup error:', err);
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
            <h1 className="text-2xl font-bold text-white mb-2">Create Account</h1>
            <p className="text-slate-400 text-sm">Join Octavia to start translating videos</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {success && (
            <div className="mb-4 p-3 bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg">
              <p className="text-accent-cyan text-sm">{success}</p>
              <p className="text-accent-cyan/70 text-xs mt-1">
                A verification link has been sent to your email.
              </p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Full Name</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="glass-input w-full !pl-12"
                  placeholder="John Doe"
                  required
                  disabled={loading}
                  minLength={2}
                />
              </div>
              <p className="text-xs text-slate-500">This will be displayed on your profile</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Email Address</label>
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
              <p className="text-xs text-slate-500">We'll send a verification link to this email</p>
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
                  minLength={6}
                  disabled={loading}
                />
              </div>
              <p className="text-xs text-slate-500">Minimum 6 characters</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Confirm Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="password"
                  value={form.confirmPassword}
                  onChange={(e) => setForm({ ...form, confirmPassword: e.target.value })}
                  className="glass-input w-full !pl-12"
                  placeholder="••••••••"
                  required
                  disabled={loading}
                />
              </div>
            </div>

            <div className="mt-6 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
              <p className="text-xs text-slate-400">
                <span className="text-blue-400 font-medium">Important:</span> After signing up, you'll receive a verification email. You must verify your email before you can log in and use Octavia.
              </p>
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
                    Creating Account...
                  </div>
                ) : (
                  "Create Account"
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
                  setError('Social login coming soon! Please use email signup for now.');
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
                  setError('Social login coming soon! Please use email signup for now.');
                }}
              >
                <span className="font-bold">G</span>
                Google
              </button>
            </div>
          </div>

          <div className="mt-8 space-y-4">
            <p className="text-center text-sm text-slate-400">
              Already have an account?{" "}
              <Link href="/login" className="text-primary-purple-bright hover:text-white transition-colors font-medium">
                Sign in
              </Link>
            </p>

            <div className="text-center">
              <Link
                href="/verify-email"
                className="inline-block text-sm text-slate-500 hover:text-slate-300 transition-colors"
              >
                Need to verify your email? Click here
              </Link>
            </div>
          </div>

          {/* Terms and Privacy */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <p className="text-xs text-center text-slate-500">
              By creating an account, you agree to our{" "}
              <Link href="/terms" className="text-slate-400 hover:text-white transition-colors">
                Terms of Service
              </Link>{" "}
              and{" "}
              <Link href="/privacy" className="text-slate-400 hover:text-white transition-colors">
                Privacy Policy
              </Link>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}