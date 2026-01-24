"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Mail, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { api } from "@/lib/api";

function VerifyEmailContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get('token');
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  // Check for email in localStorage and auto-verify if token is present
  useEffect(() => {
    // Get email from either localStorage (from signup) or URL parameters
    const storedEmail = localStorage.getItem('pending_verification_email');
    const urlEmail = searchParams.get('email');
    setEmail(urlEmail || storedEmail || '');

    // Automatically verify if token is present in URL
    if (token) {
      handleVerification(token);
    }
  }, [token]);

  // Verify email using token from verification link
  const handleVerification = async (verificationToken: string) => {
    setStatus('loading');
    try {
      const response = await api.verifyEmail(verificationToken);

      if (response.success) {
        setStatus('success');
        setMessage('Your email has been verified successfully! Redirecting to dashboard...');

        // Store user session data in localStorage
        localStorage.setItem('octavia_user', JSON.stringify({
          email: response.user.email,
          name: response.user.name,
          token: response.token,
          credits: response.user.credits,
          verified: response.user.verified
        }));

        // Clear pending email from storage
        localStorage.removeItem('pending_verification_email');

        // Redirect to dashboard after brief delay
        setTimeout(() => {
          router.push('/dashboard');
        }, 2000);
      } else {
        setStatus('error');
        setMessage(response.error || 'Verification failed. Please try again.');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Network error. Please check your connection and try again.');
    }
  };

  // Resend verification email to the user
  const handleResend = async () => {
    if (!email) return;

    setStatus('loading');
    try {
      const response = await api.resendVerification(email);
      if (response.success) {
        setStatus('success');
        setMessage('Verification email has been resent. Please check your inbox.');
      } else {
        setStatus('error');
        setMessage(response.error || 'Failed to resend email. Please try again.');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Network error. Please try again later.');
    }
  };

  return (
    <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center relative overflow-hidden">
      {/* Background glow effects */}
      <div className="glow-purple-strong"
        style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
      <div className="glow-purple"
        style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

      <div className="relative z-10 w-full max-w-md p-6">
        {/* Back to home link */}
        <Link href="/" className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        {/* Main verification panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-8"
        >
          {/* Header with logo */}
          <div className="text-center mb-8">
            <div className="w-12 h-12 mx-auto mb-4 relative flex items-center justify-center">
              <img
                src="/lunartech_logo_small.png"
                alt="LunarTech Logo"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-white/30 blur-xl rounded-full opacity-20" />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Verify Your Email</h1>
            <p className="text-slate-400 text-sm">Check your inbox to complete registration</p>
          </div>

          {/* Status and actions section */}
          <div className="space-y-6">
            <div className="text-center">
              {/* Status icon display */}
              <div className="w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                {status === 'loading' && (
                  <Loader2 className="w-16 h-16 text-primary-purple animate-spin" />
                )}
                {status === 'success' && (
                  <CheckCircle className="w-16 h-16 text-accent-cyan" />
                )}
                {status === 'error' && (
                  <XCircle className="w-16 h-16 text-red-500" />
                )}
                {status === 'idle' && (
                  <Mail className="w-16 h-16 text-primary-purple" />
                )}
              </div>

              {/* Status messages */}
              <div className="space-y-3">
                <p className="text-slate-300">
                  {status === 'idle' && (
                    `We've sent a verification link to ${email || 'your email'}. Please check your inbox and click the link to verify your account.`
                  )}
                  {status === 'loading' && 'Verifying your email...'}
                  {status === 'success' && message}
                  {status === 'error' && message}
                </p>

                {status === 'idle' && (
                  <p className="text-sm text-slate-500">
                    Didn't receive the email? Check your spam folder or click below to resend.
                  </p>
                )}
              </div>
            </div>

            {/* Action buttons */}
            <div className="space-y-3">
              {status === 'idle' && (
                <>
                  <button
                    onClick={handleResend}
                    disabled={!email}
                    className="w-full py-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-white disabled:opacity-50"
                  >
                    Resend Verification Email
                  </button>

                  {token && (
                    <button
                      onClick={() => handleVerification(token)}
                      className="w-full py-3 rounded-lg bg-primary-purple hover:bg-primary-purple-bright transition-colors text-white"
                    >
                      Click to Verify
                    </button>
                  )}
                </>
              )}

              <Link
                href="/login"
                className="block w-full py-3 text-center rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-white"
              >
                Back to Login
              </Link>
            </div>

            {/* Support information */}
            <div className="pt-6 border-t border-white/10">
              <p className="text-center text-sm text-slate-500">
                Having trouble? Contact support at{" "}
                <a href="mailto:support@octavia.com" className="text-primary-purple hover:text-white">
                  support@octavia.com
                </a>
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-purple animate-spin mx-auto mb-4" />
          <p className="text-white text-lg">Loading verification page...</p>
        </div>
      </div>
    }>
      <VerifyEmailContent />
    </Suspense>
  );
}